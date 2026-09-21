"""Orquestrador visual multi-provedor para o fluxo Raster Plan-to-BIM.

O modelo de linguagem nunca exporta IFC diretamente. Ele propõe um modelo
geométrico completo, o backend valida paredes/aberturas e o front exige uma
aprovação antes de substituir a revisão corrente.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import requests


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_BETA_CHAT_URL = "https://api.deepseek.com/beta/chat/completions"
DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_DEEPSEEK_VISION_MODEL = "deepseek-v4-flash-vision-exp"
DEFAULT_DEEPSEEK_REASONING_MODEL = "deepseek-v4-pro"
PLAN_AI_PROVIDERS = {"openai", "deepseek"}
MAX_USER_MESSAGE = 6000


class GptPlanError(RuntimeError):
    """Falha previsível do assistente visual."""


class GptPlanUnavailable(GptPlanError):
    """Integração sem credencial ou indisponível."""


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GptPlanError(f"{label} precisa ser numérico") from exc
    if not math.isfinite(number):
        raise GptPlanError(f"{label} precisa ser finito")
    return number


def _image_data_url_from_path(path: Path) -> str:
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _reference_data_url(model: dict[str, Any]) -> str:
    reference = model.get("reference") or {}
    encoded = str(reference.get("image_base64") or "")
    if not encoded:
        raise GptPlanError("O modelo não contém a imagem raster de referência.")
    mime = str(reference.get("image_mime") or "image/png")
    return f"data:{mime};base64,{encoded}"


def _decode_reference(model: dict[str, Any]) -> np.ndarray:
    reference = model.get("reference") or {}
    try:
        raw = base64.b64decode(str(reference["image_base64"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise GptPlanError("Imagem de referência inválida.") from exc
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise GptPlanError("Não foi possível decodificar a imagem de referência.")
    return image


def _model_overlay_data_url(model: dict[str, Any]) -> str:
    image = _decode_reference(model)
    height, width = image.shape[:2]
    reference = model.get("reference") or {}
    bounds = reference.get("bounds") or [
        model["bbox"]["xmin"], model["bbox"]["ymin"],
        model["bbox"]["xmax"], model["bbox"]["ymax"],
    ]
    xmin, ymin, xmax, ymax = (float(value) for value in bounds)
    span_x = max(1e-9, xmax - xmin)
    span_y = max(1e-9, ymax - ymin)

    def to_px(x: float, y: float) -> tuple[int, int]:
        return (
            int(round((float(x) - xmin) / span_x * (width - 1))),
            int(round((ymax - float(y)) / span_y * (height - 1))),
        )

    overlay = image.copy()
    contour = (model.get("laje") or {}).get("contorno") or []
    if len(contour) >= 3:
        polygon = np.asarray([to_px(*point) for point in contour], dtype=np.int32)
        cv2.fillPoly(overlay, [polygon], (210, 190, 80))
    blend = cv2.addWeighted(overlay, 0.22, image, 0.78, 0.0)
    scale = 0.5 * (width / span_x + height / span_y)
    walls = {str(wall["id"]): wall for wall in model.get("paredes", [])}
    for wall in walls.values():
        p1 = to_px(wall["ax"], wall["ay"])
        p2 = to_px(wall["bx"], wall["by"])
        thickness = max(3, int(round(float(wall["espessura"]) * scale)))
        cv2.line(blend, p1, p2, (0, 145, 255), thickness, cv2.LINE_AA)
        center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(
            blend, str(wall["id"]), center, cv2.FONT_HERSHEY_SIMPLEX,
            0.34, (80, 45, 5), 1, cv2.LINE_AA,
        )
    for opening in model.get("aberturas", []):
        wall = walls.get(str(opening.get("parede_id")))
        if not wall:
            continue
        a = np.array([float(wall["ax"]), float(wall["ay"])])
        b = np.array([float(wall["bx"]), float(wall["by"])])
        length = float(np.linalg.norm(b - a))
        if length <= 1e-9:
            continue
        unit = (b - a) / length
        center = a + unit * float(opening["s_centro"])
        half = unit * float(opening["largura"]) / 2.0
        p1, p2 = to_px(*(center - half)), to_px(*(center + half))
        color = (40, 190, 40) if opening.get("tipo") == "door" else (235, 120, 35)
        cv2.line(blend, p1, p2, color, max(5, int(float(wall["espessura"]) * scale)), cv2.LINE_AA)
        cv2.putText(
            blend, str(opening["id"]), p1, cv2.FONT_HERSHEY_SIMPLEX,
            0.32, color, 1, cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".png", blend)
    if not ok:
        raise GptPlanError("Falha ao gerar overlay para revisão visual.")
    return f"data:image/png;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


def _compact_model(model: dict[str, Any]) -> dict[str, Any]:
    reference = model.get("reference") or {}
    return {
        "nome": model.get("nome"),
        "bbox": model.get("bbox"),
        "escala_m_por_pixel": model.get("escala"),
        "reference": {
            "bounds": reference.get("bounds"),
            "canvas_size": reference.get("canvas_size"),
            "canvas_width_m": reference.get("canvas_width_m"),
        },
        "paredes": [
            {
                key: wall.get(key)
                for key in (
                    "id", "ax", "ay", "bx", "by", "espessura", "altura",
                    "layer", "nome", "tipo", "ifc_class", "confidence",
                )
            }
            for wall in model.get("paredes", [])
        ],
        "aberturas": [
            {
                key: opening.get(key)
                for key in (
                    "id", "parede_id", "tipo", "s_centro", "largura",
                    "altura", "peitoril", "nome", "confidence",
                )
            }
            for opening in model.get("aberturas", [])
        ],
        "laje": model.get("laje"),
        "metric_constraints": model.get("metric_constraints", []),
        "raster_2d": model.get("raster_2d"),
        "warnings": model.get("warnings", []),
    }


CALIBRATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "message", "needs_rectification", "source_quad_px", "main_width_m",
        "main_height_m", "right_extra_m", "dimensions", "confidence",
        "assumptions",
    ],
    "properties": {
        "message": {"type": "string"},
        "needs_rectification": {"type": "boolean"},
        "source_quad_px": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["x", "y"],
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            },
        },
        "main_width_m": {"type": ["number", "null"]},
        "main_height_m": {"type": ["number", "null"]},
        "right_extra_m": {"type": "number"},
        "dimensions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "text", "value_m", "axis", "p1_px", "p2_px",
                    "confidence", "reason",
                ],
                "properties": {
                    "text": {"type": "string"},
                    "value_m": {"type": "number"},
                    "axis": {"type": "string", "enum": ["horizontal", "vertical", "unknown"]},
                    "p1_px": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["x", "y"],
                        "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    },
                    "p2_px": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["x", "y"],
                        "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
}


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "message", "changed", "confidence", "observations", "assumptions",
        "walls", "openings", "slab_contour", "unresolved",
    ],
    "properties": {
        "message": {"type": "string"},
        "changed": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "observations": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "walls": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id", "ax", "ay", "bx", "by", "thickness", "height",
                    "kind", "name", "confidence", "reason",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "ax": {"type": "number"}, "ay": {"type": "number"},
                    "bx": {"type": "number"}, "by": {"type": "number"},
                    "thickness": {"type": "number"},
                    "height": {"type": "number"},
                    "kind": {"type": "string", "enum": ["wall", "structural-wall", "column"]},
                    "name": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
            },
        },
        "openings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id", "wall_id", "type", "s_center", "width", "height",
                    "sill", "name", "confidence", "reason",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "wall_id": {"type": "string"},
                    "type": {"type": "string", "enum": ["door", "window"]},
                    "s_center": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"},
                    "sill": {"type": "number"},
                    "name": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
            },
        },
        "slab_contour": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["x", "y"],
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            },
        },
        "unresolved": {"type": "array", "items": {"type": "string"}},
    },
}


CALIBRATION_INSTRUCTIONS = """Você é a etapa de calibração visual do protocolo Raster Plan-to-BIM.
Analise somente evidência legível. Identifique cotas arquitetônicas em metros e o quadrilátero do corpo principal da planta na imagem original, na ordem superior-esquerdo, superior-direito, inferior-direito, inferior-esquerdo.

Para CADA cota devolvida, p1_px e p2_px são obrigatoriamente os dois pontos geométricos exatos medidos pela cota na imagem original: interseções das linhas de chamada com as faces/limites medidos. Não use o centro do texto, a caixa do número nem pontos aproximados. Se o valor estiver legível mas uma das duas extremidades não puder ser localizada, não inclua essa cota.

main_width_m e main_height_m são apenas estimativas auxiliares. Não some ambientes, não invente vãos intermediários e não use uma cadeia incompleta para fabricar a dimensão externa: o backend calculará as dimensões métricas determinísticas a partir de p1_px, p2_px e do quadrilátero. right_extra_m representa uma extensão à direita, como terraço, fora do quadrilátero.

Se não houver ao menos uma cota horizontal e uma vertical com duas extremidades identificáveis, use null nas dimensões principais, array vazio no quadrilátero e explique a insuficiência. Não invente números. A confiança declarada é apenas informativa; a aplicação depende da validação geométrica local das âncoras."""


REVIEW_INSTRUCTIONS = """Você é o revisor visual do protocolo Raster Plan-to-BIM.
Receberá a planta retificada, uma segunda imagem com o resultado automático sobreposto e o JSON métrico atual. A primeira imagem é a autoridade visual; a geometria automática é apenas proposta.

Produza um MODELO COMPLETO quando changed=true, não um patch. Regras invariantes:
- primeiro consolide cada alinhamento como uma parede-mãe única e contínua;
- intervalos de portas/janelas nunca dividem a parede em duas;
- toda abertura deve apontar para um wall_id existente e caber nele;
- não transforme textos, cotas, escadas, louças ou mobiliário em paredes;
- diferencie parede grossa/estrutural de pilar compacto;
- portas exigem arco/folha ou contexto forte; janelas exigem linhas paralelas/jambas ou contexto forte;
- dúvida semântica permanece em unresolved e não cria abertura;
- preserve IDs úteis do automático; novos IDs devem ser estáveis e únicos;
- coordenadas são metros, x cresce para a direita e y para cima, dentro de reference.bounds;
- metric_constraints contém cotas e eixos protegidos pelo backend: preserve os IDs
  vinculados e nunca desloque transversalmente esses eixos de parede;
- espessuras e dimensões precisam ser arquitetonicamente plausíveis;
- laje deve acompanhar o perímetro côncavo, incluindo terraços, sem cobrir recuos indevidos.

Se o pedido for apenas uma pergunta e não exigir alteração, retorne changed=false e arrays geométricos vazios. Nunca afirme que o IFC foi exportado; a exportação depende de aprovação humana no editor."""


AUDIT_INSTRUCTIONS = """Você é o auditor geométrico final do protocolo Raster Plan-to-BIM.
Receberá o JSON métrico automático e uma proposta produzida pelo modelo visual. Não tem acesso à imagem: portanto não invente, acrescente nem remova evidência visual. Devolva um MODELO COMPLETO e preserve a proposta visual sempre que ela respeitar as regras.

Audite somente invariantes verificáveis no JSON:
- cada alinhamento físico é uma parede-mãe contínua, inclusive através de vãos;
- IDs são únicos e toda abertura referencia uma parede existente;
- centro e largura fazem a abertura caber integralmente na parede;
- espessuras, alturas, peitoris e coordenadas são finitos e plausíveis;
- pilares permanecem distintos de paredes;
- o contorno da laje tem pelo menos três pontos;
- incerteza permanece em unresolved;
- a exportação continua dependente da aprovação humana.

Se a proposta visual disser changed=false, preserve changed=false e os arrays geométricos vazios. Explique correções puramente lógicas em observations."""


class OpenAIResponsesClient:
    provider = "openai"
    requires_text_audit = False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 1200.0,
        session: Any = requests,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_PLAN_MODEL", DEFAULT_MODEL)
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = int(os.environ.get("OPENAI_PLAN_MAX_OUTPUT_TOKENS", "48000"))
        self.session = session

    def configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _output_text(payload: dict[str, Any]) -> str:
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct
        parts: list[str] = []
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(str(content["text"]))
        if not parts:
            raise GptPlanError("A API não devolveu conteúdo estruturado.")
        return "".join(parts)

    def structured(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        instructions: str,
        user_text: str,
        images: list[str],
        reasoning_effort: str | None = None,
        artifact_dir: Path | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.api_key:
            raise GptPlanUnavailable(
                "OPENAI_API_KEY não configurada no backend. Defina a variável no servidor; nunca no navegador."
            )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_text}]
        content.extend(
            {"type": "input_image", "image_url": image, "detail": "original"}
            for image in images
        )
        import time
        effort = reasoning_effort or os.environ.get("OPENAI_PLAN_REASONING_EFFORT", "medium")
        started = time.perf_counter()
        response = self.session.post(
            OPENAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "store": False,
                "instructions": instructions,
                "input": [{"role": "user", "content": content}],
                "reasoning": {"effort": effort},
                "max_output_tokens": self.max_output_tokens,
                "text": {
                    "verbosity": "low",
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            },
            timeout=self.timeout_seconds,
        )
        api_seconds = time.perf_counter() - started
        try:
            payload = response.json()
        except ValueError as exc:
            raise GptPlanError(f"Resposta inválida da OpenAI (HTTP {response.status_code}).") from exc
        if response.status_code >= 400:
            error = payload.get("error") or {}
            message = error.get("message") or f"HTTP {response.status_code}"
            raise GptPlanError(f"OpenAI: {message}")
        metadata = {
            "provider": self.provider, "response_id": payload.get("id"),
            "model": payload.get("model") or self.model, "usage": payload.get("usage"),
            "status": payload.get("status"), "incomplete_details": payload.get("incomplete_details"),
            "api_seconds": round(api_seconds, 3), "reasoning_effort": effort,
            "max_output_tokens": self.max_output_tokens, "image_count": len(images),
        }
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            # Never persist request headers or keys. Preserve partial output for diagnosis.
            for name, value in (("response.json", payload), ("api_metadata.json", metadata)):
                (artifact_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        if payload.get("status") in ("incomplete", "failed", "cancelled"):
            raise GptPlanError(f"Resposta OpenAI não concluída: {payload.get('status')}; {payload.get('incomplete_details')}")
        try:
            result = json.loads(self._output_text(payload))
        except json.JSONDecodeError as exc:
            raise GptPlanError("A saída estruturada não pôde ser interpretada.") from exc
        return result, metadata


def _deepseek_strict_schema(value: Any) -> Any:
    """Converte nosso JSON Schema para o subconjunto aceito no modo strict."""
    if isinstance(value, list):
        return [_deepseek_strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        # O strict beta aceita os tipos básicos, mas não todos os validadores
        # de coleção do JSON Schema completo.
        if key in {"maxItems", "minItems", "uniqueItems"}:
            continue
        if key == "type" and isinstance(item, list):
            non_null = [candidate for candidate in item if candidate != "null"]
            result[key] = non_null[0] if non_null else "string"
            continue
        result[key] = _deepseek_strict_schema(item)
    return result


class DeepSeekPlanClient:
    """Cliente DeepSeek mantendo toda credencial exclusivamente no backend."""

    provider = "deepseek"
    requires_text_audit = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        vision_model: str | None = None,
        reasoning_model: str | None = None,
        timeout_seconds: float = 240.0,
        session: Any = requests,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.vision_model = vision_model or os.environ.get(
            "DEEPSEEK_VISION_MODEL", DEFAULT_DEEPSEEK_VISION_MODEL
        )
        self.reasoning_model = reasoning_model or os.environ.get(
            "DEEPSEEK_REASONING_MODEL", DEFAULT_DEEPSEEK_REASONING_MODEL
        )
        audit_setting = os.environ.get("DEEPSEEK_TEXT_AUDIT", "true").strip().lower()
        self.requires_text_audit = audit_setting not in {"0", "false", "no", "off"}
        # Compatibilidade com consumidores que exibiam um único modelo.
        self.model = self.vision_model
        self.timeout_seconds = timeout_seconds
        self.session = session

    def configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _error_message(payload: dict[str, Any], status_code: int) -> str:
        error = payload.get("error") or {}
        if isinstance(error, str):
            return error
        return str(error.get("message") or f"HTTP {status_code}")

    @staticmethod
    def _message_result(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GptPlanError("A DeepSeek não devolveu uma mensagem utilizável.") from exc
        calls = message.get("tool_calls") or []
        if calls:
            raw = ((calls[0].get("function") or {}).get("arguments"))
        else:
            raw = message.get("content")
        if isinstance(raw, list):
            raw = "".join(
                str(item.get("text") or "")
                for item in raw if isinstance(item, dict)
            )
        if not isinstance(raw, str) or not raw.strip():
            raise GptPlanError("A DeepSeek devolveu conteúdo estruturado vazio.")
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GptPlanError("A saída estruturada da DeepSeek não pôde ser interpretada.") from exc
        if not isinstance(result, dict):
            raise GptPlanError("A saída estruturada da DeepSeek precisa ser um objeto JSON.")
        return result

    def _post(self, url: str, body: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        response = self.session.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout_seconds,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GptPlanError(
                f"Resposta inválida da DeepSeek (HTTP {response.status_code})."
            ) from exc
        return response, payload

    def structured(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        instructions: str,
        user_text: str,
        images: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.api_key:
            raise GptPlanUnavailable(
                "DEEPSEEK_API_KEY não configurada no backend. Defina a variável no servidor; nunca no navegador."
            )
        model = self.vision_model if images else self.reasoning_model
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        content.extend(
            {"type": "image_url", "image_url": {"url": image}}
            for image in images
        )
        strict_instructions = (
            instructions
            + "\nResponda obrigatoriamente chamando a função fornecida. "
              "Campos numéricos sem evidência devem usar 0 quando o schema strict não aceitar null."
        )
        strict_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": strict_instructions},
                {"role": "user", "content": content if images else user_text},
            ],
            "stream": False,
            "tools": [{
                "type": "function",
                "function": {
                    "name": schema_name,
                    "description": "Devolve o resultado estruturado do protocolo Plan-to-BIM.",
                    "strict": True,
                    "parameters": _deepseek_strict_schema(schema),
                },
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": schema_name},
            },
        }
        response, payload = self._post(DEEPSEEK_BETA_CHAT_URL, strict_body)
        used_fallback = False
        if response.status_code >= 400:
            # Alguns snapshots experimentais de visão ainda não aceitam strict
            # tools. Nesse caso usamos JSON mode, ainda validado localmente.
            if response.status_code not in {400, 404, 422}:
                raise GptPlanError(
                    f"DeepSeek: {self._error_message(payload, response.status_code)}"
                )
            used_fallback = True
            json_instructions = (
                instructions
                + "\nDevolva somente um objeto JSON válido que obedeça exatamente a este schema: "
                + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            )
            fallback_body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": json_instructions},
                    {"role": "user", "content": content if images else user_text},
                ],
                "stream": False,
                "response_format": {"type": "json_object"},
            }
            response, payload = self._post(DEEPSEEK_CHAT_URL, fallback_body)
            if response.status_code >= 400:
                raise GptPlanError(
                    f"DeepSeek: {self._error_message(payload, response.status_code)}"
                )
        result = self._message_result(payload)
        return result, {
            "provider": self.provider,
            "response_id": payload.get("id"),
            "model": payload.get("model") or model,
            "usage": payload.get("usage"),
            "strict_fallback": used_fallback,
        }


def _interpolate_extrapolated(value: float, xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        raise GptPlanError("Mapa métrico de cota inválido.")
    if value <= xs[0]:
        index = 0
    elif value >= xs[-1]:
        index = len(xs) - 2
    else:
        index = max(0, int(np.searchsorted(xs, value)) - 1)
    span = xs[index + 1] - xs[index]
    if span <= 1e-9:
        return ys[index]
    ratio = (value - xs[index]) / span
    return ys[index] + ratio * (ys[index + 1] - ys[index])


def _select_dimension_axis_mapping(
    records: list[dict[str, Any]], axis: str
) -> dict[str, Any] | None:
    """Escolhe uma cadeia colinear e cria uma régua 1D métrica por trechos."""
    axis_records = [item for item in records if item["axis"] == axis]
    if not axis_records:
        return None

    # Cotas da mesma cadeia ficam sobre a mesma linha de cota (coordenada
    # transversal próxima). Separar essas linhas evita misturar, por exemplo,
    # uma cadeia inferior com uma cota isolada de terraço no topo.
    groups: list[list[dict[str, Any]]] = []
    for record in sorted(axis_records, key=lambda item: item["cross"]):
        target = next(
            (
                group for group in groups
                if abs(float(np.median([item["cross"] for item in group])) - record["cross"]) <= 0.065
            ),
            None,
        )
        if target is None:
            groups.append([record])
        else:
            target.append(record)

    chains: list[list[dict[str, Any]]] = []
    for group in groups:
        ordered = sorted(group, key=lambda item: (item["start"], item["end"]))
        current: list[dict[str, Any]] = []
        current_end = -float("inf")
        for record in ordered:
            # Lacunas são permitidas: significam um trecho cuja cota não foi
            # lida e serão interpoladas pela escala local mediana. Sobreposição
            # grande, porém, indica cotas alternativas/conflitantes na mesma
            # linha; nesse caso mantemos a de maior alcance normalizado.
            if current and record["start"] < current_end - 0.035:
                if record["span"] > current[-1]["span"]:
                    current[-1] = record
                    current_end = max(item["end"] for item in current)
                continue
            current.append(record)
            current_end = max(current_end, record["end"])
        if current:
            chains.append(current)
    if not chains:
        return None

    def chain_score(chain: list[dict[str, Any]]) -> float:
        start = min(item["start"] for item in chain)
        end = max(item["end"] for item in chain)
        coverage = max(0.0, min(1.0, end) - max(0.0, start))
        boundary = (0.10 if start <= 0.04 else 0.0) + (0.10 if end >= 0.96 else 0.0)
        return coverage + boundary + min(0.24, len(chain) * 0.06)

    primary = max(chains, key=chain_score)
    primary = sorted(primary, key=lambda item: item["start"])
    local_scales = [item["value_m"] / item["span"] for item in primary]
    fallback_scale = float(np.median(local_scales))
    normalized: list[float] = [0.0]
    meters: list[float] = [0.0]
    cursor_u = 0.0
    cursor_m = 0.0
    used_indexes: list[int] = []

    for record in primary:
        start = max(cursor_u, float(record["start"]))
        if start > cursor_u + 1e-6:
            cursor_m += (start - cursor_u) * fallback_scale
            normalized.append(start)
            meters.append(cursor_m)
        end = max(start + 1e-6, float(record["end"]))
        cursor_m += float(record["value_m"])
        cursor_u = end
        if abs(normalized[-1] - end) <= 1e-6:
            meters[-1] = cursor_m
        else:
            normalized.append(end)
            meters.append(cursor_m)
        used_indexes.append(int(record["dimension_index"]))

    if cursor_u < 1.0 - 1e-6:
        cursor_m += (1.0 - cursor_u) * fallback_scale
        normalized.append(1.0)
        meters.append(cursor_m)
    elif cursor_u > 1.0 + 1e-6:
        # O corpo principal termina em u=1; interpola esse limite antes de
        # considerar possíveis extensões externas.
        main_m = _interpolate_extrapolated(1.0, normalized, meters)
        insert_at = int(np.searchsorted(normalized, 1.0))
        normalized.insert(insert_at, 1.0)
        meters.insert(insert_at, main_m)

    # Uma cota horizontal externa iniciada no limite direito amplia o canvas,
    # mas não interfere na cadeia escolhida para o corpo principal.
    if axis == "horizontal":
        main_m = _interpolate_extrapolated(1.0, normalized, meters)
        extension = max(
            (
                item for item in axis_records
                if item["start"] >= 0.96 and item["end"] > 1.02
            ),
            key=lambda item: item["end"],
            default=None,
        )
        if extension is not None:
            start_m = _interpolate_extrapolated(extension["start"], normalized, meters)
            if extension["start"] > normalized[-1] + 1e-6:
                normalized.append(float(extension["start"]))
                meters.append(start_m)
            normalized.append(float(extension["end"]))
            meters.append(start_m + float(extension["value_m"]))
            used_indexes.append(int(extension["dimension_index"]))
        # Garante que o valor em u=1 continue sendo a largura principal mesmo
        # quando a extensão foi anexada.
        _ = main_m

    return {
        "axis": axis,
        "normalized": [round(float(value), 9) for value in normalized],
        "meters": [round(float(value), 9) for value in meters],
        "dimension_indexes": sorted(set(used_indexes)),
        "method": "continuous-dimension-chain-piecewise-1d",
    }


def validate_calibration(
    calibration: dict[str, Any], image_shape: tuple[int, ...]
) -> dict[str, Any]:
    result = deepcopy(calibration)
    # O strict tool calling da DeepSeek não aceita nullable em todos os
    # snapshots; zero é o sentinela explícito instruído ao modelo.
    for key in ("main_width_m", "main_height_m"):
        try:
            if result.get(key) is not None and float(result[key]) <= 0:
                result[key] = None
        except (TypeError, ValueError):
            result[key] = None
    height, width = image_shape[:2]
    quad = result.get("source_quad_px") or []
    valid_quad = len(quad) == 4
    points: list[list[float]] = []
    if valid_quad:
        for index, item in enumerate(quad):
            x = _finite(item.get("x"), f"quad[{index}].x")
            y = _finite(item.get("y"), f"quad[{index}].y")
            if not (-0.03 * width <= x <= 1.03 * width and -0.03 * height <= y <= 1.03 * height):
                valid_quad = False
            points.append([x, y])
        if valid_quad:
            area = abs(float(cv2.contourArea(np.asarray(points, dtype=np.float32))))
            valid_quad = area >= width * height * 0.05
    model_main_width = result.get("main_width_m")
    model_main_height = result.get("main_height_m")
    result["model_main_width_m"] = model_main_width
    result["model_main_height_m"] = model_main_height

    normalized_quad = np.asarray(points, dtype=np.float32)
    unit_homography = None
    if valid_quad:
        unit_homography = cv2.getPerspectiveTransform(
            normalized_quad,
            np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        )

    dimensions: list[dict[str, Any]] = []
    extent_candidates: dict[str, list[float]] = {
        "horizontal": [],
        "vertical": [],
    }
    anchor_records: list[dict[str, Any]] = []
    for index, source_item in enumerate(result.get("dimensions", [])):
        item = deepcopy(source_item)
        try:
            value_m = float(item.get("value_m", 0))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(value_m) and 0.05 <= value_m <= 500):
            continue
        axis = str(item.get("axis") or "unknown")
        p1 = item.get("p1_px")
        p2 = item.get("p2_px")
        anchor_valid = bool(
            unit_homography is not None
            and axis in extent_candidates
            and isinstance(p1, dict)
            and isinstance(p2, dict)
        )
        source_points: list[list[float]] = []
        if anchor_valid:
            try:
                for label, point in (("p1", p1), ("p2", p2)):
                    x = _finite(point.get("x"), f"dimension[{index}].{label}.x")
                    y = _finite(point.get("y"), f"dimension[{index}].{label}.y")
                    if not (
                        -0.03 * width <= x <= 1.03 * width
                        and -0.03 * height <= y <= 1.03 * height
                    ):
                        anchor_valid = False
                    source_points.append([x, y])
            except GptPlanError:
                anchor_valid = False
        if anchor_valid:
            projected = cv2.perspectiveTransform(
                np.asarray([source_points], dtype=np.float32), unit_homography
            )[0]
            delta_x = abs(float(projected[1, 0] - projected[0, 0]))
            delta_y = abs(float(projected[1, 1] - projected[0, 1]))
            along = delta_x if axis == "horizontal" else delta_y
            cross = delta_y if axis == "horizontal" else delta_x
            anchor_valid = bool(
                math.isfinite(along)
                and along >= 0.015
                and cross <= max(0.06, along * 0.30)
            )
            if anchor_valid:
                extent_m = value_m / along
                anchor_valid = 0.5 <= extent_m <= 500
                if anchor_valid:
                    extent_candidates[axis].append(extent_m)
                    item["normalized_p1"] = [
                        round(float(projected[0, 0]), 8),
                        round(float(projected[0, 1]), 8),
                    ]
                    item["normalized_p2"] = [
                        round(float(projected[1, 0]), 8),
                        round(float(projected[1, 1]), 8),
                    ]
                    item["extent_candidate_m"] = round(extent_m, 8)
                    along_values = (
                        [float(projected[0, 0]), float(projected[1, 0])]
                        if axis == "horizontal"
                        else [float(projected[0, 1]), float(projected[1, 1])]
                    )
                    cross_values = (
                        [float(projected[0, 1]), float(projected[1, 1])]
                        if axis == "horizontal"
                        else [float(projected[0, 0]), float(projected[1, 0])]
                    )
                    anchor_records.append({
                        "dimension_index": len(dimensions),
                        "axis": axis,
                        "start": min(along_values),
                        "end": max(along_values),
                        "span": along,
                        "cross": sum(cross_values) / 2.0,
                        "value_m": value_m,
                    })
        item["anchor_valid"] = anchor_valid
        dimensions.append(item)

    horizontal_mapping = _select_dimension_axis_mapping(anchor_records, "horizontal")
    vertical_mapping = _select_dimension_axis_mapping(anchor_records, "vertical")
    axis_mappings = {
        "horizontal": horizontal_mapping,
        "vertical": vertical_mapping,
    }
    main_width = (
        _interpolate_extrapolated(
            1.0, horizontal_mapping["normalized"], horizontal_mapping["meters"]
        )
        if horizontal_mapping else None
    )
    main_height = (
        _interpolate_extrapolated(
            1.0, vertical_mapping["normalized"], vertical_mapping["meters"]
        )
        if vertical_mapping else None
    )
    used_indexes = {
        int(index)
        for mapping in axis_mappings.values() if mapping
        for index in mapping["dimension_indexes"]
    }
    for dimension_index, item in enumerate(dimensions):
        if not item.get("anchor_valid"):
            continue
        mapping = axis_mappings.get(str(item.get("axis")))
        if mapping is None:
            continue
        normalized_p1 = item["normalized_p1"]
        normalized_p2 = item["normalized_p2"]
        first = (
            float(normalized_p1[0])
            if item.get("axis") == "horizontal"
            else float(normalized_p1[1])
        )
        second = (
            float(normalized_p2[0])
            if item.get("axis") == "horizontal"
            else float(normalized_p2[1])
        )
        measured_m = abs(
            _interpolate_extrapolated(second, mapping["normalized"], mapping["meters"])
            - _interpolate_extrapolated(first, mapping["normalized"], mapping["meters"])
        )
        item["derived_value_m"] = round(measured_m, 6)
        item["relative_error"] = round(
            abs(measured_m - float(item["value_m"])) / float(item["value_m"]),
            6,
        )
        item["used_for_scale"] = dimension_index in used_indexes

    non_parallel = bool(
        horizontal_mapping and vertical_mapping
    )
    physical = (
        main_width is not None and main_height is not None
        and 0.5 <= float(main_width) <= 500
        and 0.5 <= float(main_height) <= 500
    )
    result["source_quad_px"] = points if valid_quad else []
    result["dimensions"] = dimensions
    result["axis_mappings"] = axis_mappings
    result["main_width_m"] = round(main_width, 8) if main_width is not None else None
    result["main_height_m"] = round(main_height, 8) if main_height is not None else None
    if horizontal_mapping and max(horizontal_mapping["normalized"]) > 1.0:
        furthest_u = max(horizontal_mapping["normalized"])
        right_edge_m = _interpolate_extrapolated(
            furthest_u, horizontal_mapping["normalized"], horizontal_mapping["meters"]
        )
        result["right_extra_m"] = round(max(0.0, right_edge_m - float(main_width)), 8)
    result["applied"] = bool(
        result.get("needs_rectification")
        and valid_quad and non_parallel and physical
    )
    result["validation"] = {
        "valid_quad": valid_quad,
        "reported_dimensions": len(dimensions),
        "anchored_dimensions": sum(bool(item.get("anchor_valid")) for item in dimensions),
        "horizontal_anchor_candidates": len(extent_candidates["horizontal"]),
        "vertical_anchor_candidates": len(extent_candidates["vertical"]),
        "scale_dimensions": len(used_indexes),
        "non_parallel_dimensions": non_parallel,
        "physical_extent_valid": physical,
        "extent_source": "continuous-dimension-chain-piecewise-1d" if physical else "none",
        "model_confidence_advisory_only": True,
    }
    return result


def _wall_axis(wall: dict[str, Any]) -> tuple[str, float, float, float]:
    ax = float(wall.get("ax", 0.0))
    ay = float(wall.get("ay", 0.0))
    bx = float(wall.get("bx", 0.0))
    by = float(wall.get("by", 0.0))
    if abs(by - ay) >= abs(bx - ax):
        return "vertical", (ax + bx) / 2.0, min(ay, by), max(ay, by)
    return "horizontal", (ay + by) / 2.0, min(ax, bx), max(ax, bx)


def _nearest_wall_for_dimension_point(
    walls: list[dict[str, Any]],
    point: list[float],
    *,
    orientation: str,
    excluded_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    excluded_ids = excluded_ids or set()
    along_coordinate = float(point[1] if orientation == "vertical" else point[0])
    fixed_coordinate = float(point[0] if orientation == "vertical" else point[1])
    candidates: list[tuple[float, float, dict[str, Any]]] = []
    for wall in walls:
        identifier = str(wall.get("id") or "")
        if identifier in excluded_ids or str(wall.get("tipo") or "") == "column":
            continue
        wall_orientation, fixed, start, end = _wall_axis(wall)
        if wall_orientation != orientation:
            continue
        along_gap = max(0.0, start - along_coordinate, along_coordinate - end)
        fixed_gap = abs(fixed - fixed_coordinate)
        candidates.append((fixed_gap + along_gap * 0.08, fixed_gap, wall))
    if not candidates:
        return None
    _score, fixed_gap, wall = min(candidates, key=lambda item: item[0])
    thickness = float(wall.get("espessura", wall.get("thickness", 0.15)) or 0.15)
    if fixed_gap > max(0.75, thickness * 3.0):
        return None
    return wall


def build_metric_constraints(
    calibration: dict[str, Any],
    rectification: dict[str, Any],
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    """Projeta cotas ancoradas no canvas métrico e liga seus eixos às paredes 2D."""
    homography = np.asarray(rectification["homography"], dtype=np.float64)
    unit_homography = np.asarray(
        rectification.get("unit_homography") or [], dtype=np.float64
    )
    axis_mappings = rectification.get("axis_mappings") or {}
    pixels_per_meter = float(rectification["pixels_per_meter"])
    image_width = int(round(float(rectification["canvas_width_m"]) * pixels_per_meter))
    image_height = int(round(float(rectification["canvas_height_m"]) * pixels_per_meter))
    canvas_size = int((model.get("reference") or {}).get("canvas_size", [max(image_width, image_height)])[0])
    pixel_m = float(model.get("escala") or (float(model["bbox"]["xmax"]) / canvas_size))
    pad_x = (canvas_size - image_width) // 2
    pad_y = (canvas_size - image_height) // 2
    walls = list(model.get("paredes") or [])
    constraints: list[dict[str, Any]] = []

    def world_point(source_point: dict[str, Any]) -> list[float]:
        if unit_homography.shape == (3, 3) and axis_mappings:
            unit_point = cv2.perspectiveTransform(
                np.asarray([[[float(source_point["x"]), float(source_point["y"])]]], dtype=np.float64),
                unit_homography,
            )[0, 0]
            horizontal = axis_mappings["horizontal"]
            vertical = axis_mappings["vertical"]
            metric_x = _interpolate_extrapolated(
                float(unit_point[0]), horizontal["normalized"], horizontal["meters"]
            )
            metric_y = _interpolate_extrapolated(
                float(unit_point[1]), vertical["normalized"], vertical["meters"]
            )
            projected = np.asarray([
                (metric_x + float(rectification["margin_m"])) * pixels_per_meter,
                (metric_y + float(rectification["margin_m"])) * pixels_per_meter,
            ])
        else:
            projected = cv2.perspectiveTransform(
                np.asarray([[[float(source_point["x"]), float(source_point["y"])]]], dtype=np.float64),
                homography,
            )[0, 0]
        return [
            round((float(projected[0]) + pad_x) * pixel_m, 6),
            round((canvas_size - (float(projected[1]) + pad_y)) * pixel_m, 6),
        ]

    for index, dimension in enumerate(calibration.get("dimensions") or [], 1):
        if not dimension.get("anchor_valid"):
            continue
        hard_constraint = bool(dimension.get("used_for_scale"))
        axis = str(dimension["axis"])
        p1_world = world_point(dimension["p1_px"])
        p2_world = world_point(dimension["p2_px"])
        measured_m = (
            abs(p2_world[0] - p1_world[0])
            if axis == "horizontal"
            else abs(p2_world[1] - p1_world[1])
        )
        wall_orientation = "vertical" if axis == "horizontal" else "horizontal"
        first_wall = (
            _nearest_wall_for_dimension_point(walls, p1_world, orientation=wall_orientation)
            if hard_constraint else None
        )
        excluded = {str(first_wall["id"])} if first_wall else set()
        second_wall = (
            _nearest_wall_for_dimension_point(
                walls, p2_world, orientation=wall_orientation, excluded_ids=excluded
            )
            if hard_constraint else None
        )
        bindings: list[dict[str, Any]] = []
        for role, wall in (("p1", first_wall), ("p2", second_wall)):
            if wall is None:
                continue
            orientation, fixed, _start, _end = _wall_axis(wall)
            bindings.append({
                "role": role,
                "wall_id": str(wall["id"]),
                "orientation": orientation,
                "locked_axis_m": round(fixed, 6),
            })
        constraints.append({
            "id": f"DIM-{index:03d}",
            "text": str(dimension.get("text") or dimension["value_m"]),
            "value_m": round(float(dimension["value_m"]), 6),
            "axis": axis,
            "hard_constraint": hard_constraint,
            "p1_source_px": [
                round(float(dimension["p1_px"]["x"]), 3),
                round(float(dimension["p1_px"]["y"]), 3),
            ],
            "p2_source_px": [
                round(float(dimension["p2_px"]["x"]), 3),
                round(float(dimension["p2_px"]["y"]), 3),
            ],
            "p1_world": p1_world,
            "p2_world": p2_world,
            "rectified_measure_m": round(measured_m, 6),
            "relative_error": round(
                abs(measured_m - float(dimension["value_m"]))
                / float(dimension["value_m"]),
                6,
            ),
            "wall_bindings": bindings,
        })
    return constraints


def _wall_as_review_item(wall: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(wall["id"]),
        "ax": float(wall["ax"]), "ay": float(wall["ay"]),
        "bx": float(wall["bx"]), "by": float(wall["by"]),
        "thickness": float(wall.get("espessura", 0.15)),
        "height": float(wall.get("altura", 2.8)),
        "kind": str(wall.get("tipo") or "wall"),
        "name": str(wall.get("nome") or wall["id"]),
        "confidence": float(wall.get("confidence", 0.5)),
        "reason": "eixo restaurado por cota métrica ancorada",
    }


def enforce_metric_constraints(
    current: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any]:
    """Impede a revisão semântica de deslocar eixos ligados a cotas impressas."""
    constrained = deepcopy(review)
    constraints = list(current.get("metric_constraints") or [])
    if not constrained.get("changed") or not constraints:
        return constrained

    protected: dict[str, dict[str, Any]] = {}
    for constraint in constraints:
        for binding in constraint.get("wall_bindings") or []:
            identifier = str(binding.get("wall_id") or "")
            if identifier:
                protected[identifier] = {
                    "orientation": str(binding["orientation"]),
                    "fixed": float(binding["locked_axis_m"]),
                }
    if not protected:
        return constrained

    original_walls = {
        str(wall["id"]): wall for wall in current.get("paredes") or []
    }
    proposed = list(constrained.get("walls") or [])
    openings = list(constrained.get("openings") or [])
    used_indexes: set[int] = set()
    restored: list[str] = []
    renamed: list[dict[str, str]] = []

    for identifier, lock in protected.items():
        match_index = next(
            (index for index, wall in enumerate(proposed) if str(wall.get("id")) == identifier),
            None,
        )
        if match_index is None:
            candidates: list[tuple[float, int]] = []
            for index, wall in enumerate(proposed):
                if index in used_indexes:
                    continue
                orientation, fixed, _start, _end = _wall_axis(wall)
                if orientation == lock["orientation"]:
                    candidates.append((abs(fixed - lock["fixed"]), index))
            if candidates and min(candidates)[0] <= 0.75:
                _distance, match_index = min(candidates)
                previous_id = str(proposed[match_index].get("id") or "")
                proposed[match_index]["id"] = identifier
                for opening in openings:
                    if str(opening.get("wall_id") or "") == previous_id:
                        opening["wall_id"] = identifier
                renamed.append({"from": previous_id, "to": identifier})
            else:
                original = original_walls.get(identifier)
                if original is None:
                    continue
                proposed.append(_wall_as_review_item(original))
                match_index = len(proposed) - 1
                restored.append(identifier)
        used_indexes.add(match_index)
        wall = proposed[match_index]
        if lock["orientation"] == "vertical":
            wall["ax"] = lock["fixed"]
            wall["bx"] = lock["fixed"]
        else:
            wall["ay"] = lock["fixed"]
            wall["by"] = lock["fixed"]

    constrained["walls"] = proposed
    constrained["openings"] = openings
    constrained.setdefault("observations", []).append(
        f"Backend preservou {len(protected)} eixo(s) de parede vinculados a cotas ancoradas."
    )
    constrained["_metric_constraint_enforcement"] = {
        "protected_wall_axes": len(protected),
        "restored_wall_ids": restored,
        "renamed_wall_ids": renamed,
    }
    return constrained


def build_candidate_model(
    current: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any]:
    review = enforce_metric_constraints(current, review)
    provider = str(review.get("_provider") or "openai").lower()
    openai_model = str(review.get("_model") or DEFAULT_MODEL)
    source_label = "deepseek-v4" if provider == "deepseek" else openai_model
    review_label = "DeepSeek V4" if provider == "deepseek" else (
        "GPT-6 Astra" if openai_model.startswith("gpt-6-astra") else openai_model
    )
    if not review.get("changed"):
        candidate = deepcopy(current)
        candidate["gpt_plan"] = deepcopy(review)
        return candidate
    proposed_walls = review.get("walls") or []
    if not proposed_walls:
        raise GptPlanError("A revisão marcou mudança, mas não devolveu paredes.")
    if len(proposed_walls) > 500 or len(review.get("openings") or []) > 1000:
        raise GptPlanError("A revisão excedeu o limite de elementos.")
    bbox = current.get("bbox") or {"xmin": 0, "ymin": 0, "xmax": 20, "ymax": 20}
    xmin, ymin = float(bbox["xmin"]), float(bbox["ymin"])
    xmax, ymax = float(bbox["xmax"]), float(bbox["ymax"])
    margin = max(xmax - xmin, ymax - ymin, 1.0) * 0.15

    ids: set[str] = set()
    walls: list[dict[str, Any]] = []
    lengths: dict[str, float] = {}
    for index, item in enumerate(proposed_walls, 1):
        identifier = str(item.get("id") or f"W-GPT-{index:03d}").strip()
        if not identifier or identifier in ids:
            raise GptPlanError(f"ID de parede duplicado ou vazio: {identifier!r}")
        ids.add(identifier)
        ax = _finite(item.get("ax"), f"{identifier}.ax")
        ay = _finite(item.get("ay"), f"{identifier}.ay")
        bx = _finite(item.get("bx"), f"{identifier}.bx")
        by = _finite(item.get("by"), f"{identifier}.by")
        if not all((xmin - margin <= value <= xmax + margin) for value in (ax, bx)):
            raise GptPlanError(f"Parede {identifier} saiu dos limites X do canvas.")
        if not all((ymin - margin <= value <= ymax + margin) for value in (ay, by)):
            raise GptPlanError(f"Parede {identifier} saiu dos limites Y do canvas.")
        length = math.hypot(bx - ax, by - ay)
        if length < 0.05:
            raise GptPlanError(f"Parede degenerada: {identifier}")
        thickness = _finite(item.get("thickness"), f"{identifier}.thickness")
        if not 0.03 <= thickness <= 1.50:
            raise GptPlanError(f"Espessura implausível em {identifier}: {thickness}")
        kind = str(item.get("kind") or "wall")
        column = kind == "column"
        walls.append({
            "id": identifier,
            "ax": round(ax, 5), "ay": round(ay, 5),
            "bx": round(bx, 5), "by": round(by, 5),
            "espessura": round(thickness, 4),
            "altura": round(max(0.1, _finite(item.get("height", 2.8), f"{identifier}.height")), 4),
            "elevacao": 0.0,
            "layer": "Column-GPT-Visual" if column else (
                "Wall-Structural-GPT-Visual" if kind == "structural-wall" else "Wall-GPT-Visual"
            ),
            "nome": str(item.get("name") or identifier),
            "tipo": kind,
            "ifc_class": "IfcColumn" if column else "IfcWall",
            "origem": f"{source_label}-visual-review",
            "confidence": round(min(1.0, max(0.0, float(item.get("confidence", 0.5)))), 4),
            "semantic_reason": str(item.get("reason") or "revisão visual"),
        })
        lengths[identifier] = length

    opening_ids: set[str] = set()
    openings: list[dict[str, Any]] = []
    for index, item in enumerate(review.get("openings") or [], 1):
        identifier = str(item.get("id") or f"O-GPT-{index:03d}").strip()
        if not identifier or identifier in opening_ids or identifier in ids:
            raise GptPlanError(f"ID de abertura duplicado ou vazio: {identifier!r}")
        opening_ids.add(identifier)
        wall_id = str(item.get("wall_id") or "")
        if wall_id not in lengths:
            raise GptPlanError(f"Abertura {identifier} sem parede hospedeira válida.")
        width = _finite(item.get("width"), f"{identifier}.width")
        center = _finite(item.get("s_center"), f"{identifier}.s_center")
        wall_length = lengths[wall_id]
        if width < 0.20 or width > wall_length - 0.02:
            raise GptPlanError(f"Abertura {identifier} não cabe em {wall_id}.")
        if center - width / 2 < -1e-5 or center + width / 2 > wall_length + 1e-5:
            raise GptPlanError(f"Centro da abertura {identifier} saiu de {wall_id}.")
        kind = str(item.get("type"))
        sill = max(0.0, _finite(item.get("sill", 0), f"{identifier}.sill"))
        if kind == "door":
            sill = 0.0
        openings.append({
            "id": identifier,
            "parede_id": wall_id,
            "tipo": kind,
            "s_centro": round(center, 5),
            "largura": round(width, 4),
            "altura": round(max(0.20, _finite(item.get("height"), f"{identifier}.height")), 4),
            "peitoril": round(sill, 4),
            "nome": str(item.get("name") or identifier),
            "origem": f"{source_label}-visual-review",
            "confidence": round(min(1.0, max(0.0, float(item.get("confidence", 0.5)))), 4),
            "semantic_reason": str(item.get("reason") or "revisão visual"),
        })

    slab = []
    for index, point in enumerate(review.get("slab_contour") or []):
        x = _finite(point.get("x"), f"slab[{index}].x")
        y = _finite(point.get("y"), f"slab[{index}].y")
        if not (xmin - margin <= x <= xmax + margin and ymin - margin <= y <= ymax + margin):
            raise GptPlanError("O contorno proposto da laje saiu do canvas.")
        slab.append([round(x, 5), round(y, 5)])
    if len(slab) < 3:
        slab = deepcopy((current.get("laje") or {}).get("contorno") or [])

    candidate = deepcopy(current)
    candidate["paredes"] = walls
    candidate["aberturas"] = openings
    candidate["laje"] = {
        "contorno": slab,
        "piso": deepcopy((current.get("laje") or {}).get("piso") or {"ativo": True, "espessura": 0.12}),
        "teto": deepcopy((current.get("laje") or {}).get("teto") or {"ativo": False, "espessura": 0.12}),
    }
    candidate["spaces"] = []
    candidate["diagnostico"] = {
        **(current.get("diagnostico") or {}),
        "blocos_esquadria": len(openings),
        "elementos_lidos": len(walls) + len(openings) + (1 if len(slab) >= 3 else 0),
    }
    source = deepcopy(current.get("source") or {})
    source["mode"] = f"morphology-2d+{source_label}-vision"
    source["semantic_level"] = "visual-reviewed-geometric"
    candidate["source"] = source
    candidate["warnings"] = [
        *(current.get("warnings") or []),
        f"Revisão visual {review_label} proposta; confirme-a no editor antes de exportar IFC/DXF.",
        *(f"PENDÊNCIA IA: {item}" for item in review.get("unresolved") or []),
    ]
    candidate["gpt_plan"] = deepcopy(review)
    return candidate


class GptPlanAssistant:
    def __init__(
        self,
        client: Any | None = None,
        *,
        provider: str | None = None,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            selected_provider = (
                provider or os.environ.get("PLAN_AI_PROVIDER", "openai")
            ).strip().lower()
            if selected_provider not in PLAN_AI_PROVIDERS:
                raise GptPlanUnavailable(
                    f"Provedor de IA desconhecido: {selected_provider!r}. Use openai ou deepseek."
                )
            if selected_provider == "deepseek":
                self.client = DeepSeekPlanClient()
            else:
                self.client = OpenAIResponsesClient()

    def status(self) -> dict[str, Any]:
        openai_client = OpenAIResponsesClient()
        deepseek_client = DeepSeekPlanClient()
        deepseek_workflow = (
            "vision-calibration -> local-2d -> vision-review -> "
            "text-audit -> human-approval"
            if deepseek_client.requires_text_audit
            else "vision-calibration -> local-2d -> vision-review -> human-approval"
        )
        return {
            "configured": self.client.configured(),
            "provider": self.client.provider,
            "default_provider": self.client.provider,
            "model": self.client.model,
            "vision_model": getattr(self.client, "vision_model", self.client.model),
            "reasoning_model": getattr(self.client, "reasoning_model", self.client.model),
            "workflow": (
                "vision-calibration -> local-2d -> vision-review -> "
                "text-audit -> human-approval"
                if getattr(self.client, "requires_text_audit", False)
                else "vision-calibration -> local-2d -> vision-review -> human-approval"
            ),
            "providers": {
                "openai": {
                    "configured": openai_client.configured(),
                    "label": openai_client.model,
                    "model": openai_client.model,
                    "workflow": "vision-calibration -> local-2d -> vision-review -> human-approval",
                },
                "deepseek": {
                    "configured": deepseek_client.configured(),
                    "label": "DeepSeek V4",
                    "model": deepseek_client.vision_model,
                    "vision_model": deepseek_client.vision_model,
                    "reasoning_model": deepseek_client.reasoning_model,
                    "workflow": deepseek_workflow,
                },
            },
            "stores_responses": False,
        }

    def analyze_calibration(
        self, image_path: Path, user_message: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise GptPlanError("Não foi possível abrir a planta enviada.")
        prompt = (
            f"Pedido do usuário: {user_message[:MAX_USER_MESSAGE]}\n"
            f"Dimensões exatas da imagem: {image.shape[1]} × {image.shape[0]} pixels. "
            "Analise a escala e a necessidade de retificação."
        )
        result, metadata = self.client.structured(
            schema_name="plan_bim_calibration",
            schema=CALIBRATION_SCHEMA,
            instructions=CALIBRATION_INSTRUCTIONS,
            user_text=prompt,
            images=[_image_data_url_from_path(image_path)],
        )
        result["_provider"] = self.client.provider
        result["_model"] = metadata.get("model") or self.client.model
        return validate_calibration(result, image.shape), metadata

    def review_model(
        self, model: dict[str, Any], user_message: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        visual_started_at = time.perf_counter()
        compact = _compact_model(model)
        prompt = (
            "Imagem 1: raster retificado/original usado como referência.\n"
            "Imagem 2: resultado automático, com paredes laranja, portas verdes, "
            "janelas azuis e IDs desenhados.\n"
            f"Pedido do usuário: {user_message[:MAX_USER_MESSAGE]}\n"
            "JSON métrico atual:\n"
            + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        )
        visual_result, visual_metadata = self.client.structured(
            schema_name="plan_bim_visual_review",
            schema=REVIEW_SCHEMA,
            instructions=REVIEW_INSTRUCTIONS,
            user_text=prompt,
            images=[_reference_data_url(model), _model_overlay_data_url(model)],
        )
        print(
            f"[plan-to-bim] provider={self.client.provider} "
            f"stage=vision-review elapsed_s={time.perf_counter() - visual_started_at:.2f}",
            flush=True,
        )
        visual_result["_provider"] = self.client.provider
        visual_result["_model"] = visual_metadata.get("model") or self.client.model
        if not getattr(self.client, "requires_text_audit", False):
            return visual_result, visual_metadata

        audit_started_at = time.perf_counter()
        audit_prompt = (
            f"Pedido original do usuário: {user_message[:MAX_USER_MESSAGE]}\n"
            "JSON métrico automático:\n"
            + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
            + "\nProposta do modelo visual:\n"
            + json.dumps(visual_result, ensure_ascii=False, separators=(",", ":"))
        )
        audited_result, audit_metadata = self.client.structured(
            schema_name="plan_bim_geometry_audit",
            schema=REVIEW_SCHEMA,
            instructions=AUDIT_INSTRUCTIONS,
            user_text=audit_prompt,
            images=[],
        )
        print(
            f"[plan-to-bim] provider={self.client.provider} "
            f"stage=text-audit elapsed_s={time.perf_counter() - audit_started_at:.2f}",
            flush=True,
        )
        audited_result["_provider"] = self.client.provider
        return audited_result, {
            "provider": self.client.provider,
            "vision": visual_metadata,
            "audit": audit_metadata,
        }

    def run_initial(
        self,
        image_path: Path,
        job_dir: Path,
        *,
        fallback_canvas_width_m: float,
        user_message: str,
    ) -> dict[str, Any]:
        from .raster_2d_import import raster_2d_image_to_editor_model
        from .rectify_scaled_floorplan import rectify_floorplan

        total_started_at = time.perf_counter()
        calibration_started_at = time.perf_counter()
        calibration, calibration_api = self.analyze_calibration(image_path, user_message)
        print(
            f"[plan-to-bim] provider={self.client.provider} "
            f"stage=vision-calibration elapsed_s={time.perf_counter() - calibration_started_at:.2f}",
            flush=True,
        )
        work_image = image_path
        canvas_width_m = float(fallback_canvas_width_m)
        rectification = None
        if calibration.get("applied"):
            work_image = job_dir / "rectified_scaled.png"
            quad = np.asarray(calibration["source_quad_px"], dtype=np.float32)
            rectification = rectify_floorplan(
                image_path,
                work_image,
                source_quad=quad,
                main_width_m=float(calibration["main_width_m"]),
                main_height_m=float(calibration["main_height_m"]),
                pixels_per_meter=100.0,
                margin_m=0.5,
                right_extra_m=max(0.0, min(50.0, float(calibration.get("right_extra_m", 0.0)))),
                normalize=True,
                x_mapping=(calibration.get("axis_mappings") or {}).get("horizontal"),
                y_mapping=(calibration.get("axis_mappings") or {}).get("vertical"),
            )
            canvas_width_m = float(rectification["canvas_width_m"])

        local_2d_started_at = time.perf_counter()
        model = raster_2d_image_to_editor_model(
            work_image,
            canvas_width_m=canvas_width_m,
        )
        print(
            f"[plan-to-bim] provider={self.client.provider} "
            f"stage=local-2d elapsed_s={time.perf_counter() - local_2d_started_at:.2f}",
            flush=True,
        )
        model["calibration"] = deepcopy(calibration)
        if rectification is not None:
            model["metric_constraints"] = build_metric_constraints(
                calibration, rectification, model
            )
            model.setdefault("source", {})["scale_source"] = "anchored-dimensions"
            model.setdefault("reference", {})["metric_constraints"] = deepcopy(
                model["metric_constraints"]
            )
        else:
            model["metric_constraints"] = []
        review_started_at = time.perf_counter()
        review, review_api = self.review_model(model, user_message)
        print(
            f"[plan-to-bim] provider={self.client.provider} "
            f"stage=review-total elapsed_s={time.perf_counter() - review_started_at:.2f} "
            f"total_s={time.perf_counter() - total_started_at:.2f}",
            flush=True,
        )
        candidate = build_candidate_model(model, review)
        effective_review = candidate.get("gpt_plan") or review
        report = {
            "assistant": str(effective_review.get("message") or calibration.get("message") or "Análise concluída."),
            "calibration": calibration,
            "rectification": rectification,
            "metric_constraints": deepcopy(model.get("metric_constraints") or []),
            "review": effective_review,
            "api": {"calibration": calibration_api, "review": review_api},
            "model": candidate,
        }
        (job_dir / "gpt_plan_report.json").write_text(
            json.dumps({key: value for key, value in report.items() if key != "model"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report
