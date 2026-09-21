"""MCP-ready descriptions for the deterministic editing engine."""

from __future__ import annotations


SUPPORTED_OPERATIONS = [
    {
        "op": "delete_elements",
        "purpose": "Remove paredes ou aberturas por ID.",
        "required": ["ids"],
    },
    {
        "op": "add_wall",
        "purpose": "Cria uma parede por pontos ou por direção até outro eixo.",
        "required": ["id", "from", "direction", "until"],
    },
    {
        "op": "move_wall_endpoint",
        "purpose": "Move P1/P2 para coordenada, seletor ou interseção.",
        "required": ["selector", "target"],
    },
    {
        "op": "connect_endpoint",
        "purpose": "Alias explícito para conectar uma ponta sem trocar o eixo.",
        "required": ["selector", "target"],
    },
    {
        "op": "move_wall",
        "purpose": "Translada a parede inteira.",
        "required": ["id", "delta"],
    },
    {
        "op": "set_wall_thickness",
        "purpose": "Altera a espessura física da parede.",
        "required": ["id", "thickness"],
    },
    {
        "op": "merge_walls",
        "purpose": "Mescla volumes colineares e mede a espessura física total.",
        "required": ["ids"],
    },
    {
        "op": "add_opening",
        "purpose": "Adiciona porta ou janela hospedada numa parede.",
        "required": ["id", "wall_id", "type", "width"],
    },
    {
        "op": "move_opening",
        "purpose": "Move uma abertura ao longo do eixo hospedeiro.",
        "required": ["id", "s_center"],
    },
    {
        "op": "resize_opening",
        "purpose": "Altera a largura de uma abertura.",
        "required": ["id", "width"],
    },
    {
        "op": "set_opening_type",
        "purpose": (
            "Reclassifica porta/janela preservando, por padrão, a cota superior "
            "observada quando uma janela vira porta."
        ),
        "required": ["id", "type"],
    },
    {
        "op": "copy_opening_pattern",
        "purpose": (
            "Projeta perpendicularmente um padrão de portas/janelas entre "
            "paredes paralelas."
        ),
        "required": ["source_wall_id", "target_wall_id"],
    },
    {
        "op": "close_wall_junctions",
        "purpose": (
            "Fecha encontros L e T preservando os eixos e evitando aberturas."
        ),
        "required": ["max_distance"],
    },
    {
        "op": "close_small_gaps",
        "purpose": "Fecha encontros perpendiculares abaixo de uma tolerância explícita.",
        "required": ["max_distance"],
    },
]


def _describe_legacy_mcp_surface() -> dict:
    return {
        "schema": "bim.editing-mcp-surface.v1",
        "resources": [
            {
                "uri": "bim://editing/operations",
                "description": "Operações determinísticas aceitas pelo motor.",
            },
            {
                "uri": "bim://project/{project_id}/revision/{revision_id}/parts",
                "description": "Seletores estáveis como W-005.P1 e W-005.P2.",
            },
        ],
        "tools": [
            {
                "name": "apply_bim_revision",
                "description": (
                    "Aplica operações declarativas a um modelo, recalcula "
                    "dependências e devolve uma nova revisão sem alterar a base."
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["model", "revision"],
                    "properties": {
                        "model": {"type": "object"},
                        "revision": {
                            "type": "object",
                            "required": ["operations"],
                        },
                    },
                },
            },
            {
                "name": "resolve_bim_part",
                "description": "Resolve um seletor de parede, ponta ou eixo.",
                "input_schema": {
                    "type": "object",
                    "required": ["model", "selector"],
                    "properties": {
                        "model": {"type": "object"},
                        "selector": {
                            "type": "string",
                            "examples": ["W-S01-005.P1", "W-S01-005.AXIS"],
                        },
                    },
                },
            },
        ],
        "operations": SUPPORTED_OPERATIONS,
        "language_model_required": False,
    }


def describe_mcp_surface() -> dict:
    """Describe the executable, file-based MCP workflow."""

    nullable_string = {"type": ["string", "null"]}
    nullable_selection = {
        "type": ["array", "null"],
        "items": {"type": "string"},
    }
    return {
        "schema": "bim.editing-mcp-surface.v2",
        "resources": [
            {
                "uri": "bim://editing/operations",
                "description": "Operacoes deterministicas aceitas pelo motor.",
            },
            {
                "uri": "bim://editing/workflow",
                "description": (
                    "Fluxo obrigatorio IFC -> JSON -> revisao -> PNG -> "
                    "aprovacao -> IFC."
                ),
            },
        ],
        "tools": [
            {
                "name": "recover_ifc_for_editing",
                "description": (
                    "Recupera IFC no editor JSON e devolve caminhos dos "
                    "artefatos de revisao."
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["ifc_path"],
                    "properties": {
                        "ifc_path": {"type": "string"},
                        "output_dir": nullable_string,
                        "force_ceiling": {"type": "boolean"},
                        "selected": nullable_selection,
                    },
                },
            },
            {
                "name": "resolve_bim_part",
                "description": "Resolve seletor estavel como W-S01-005.P1.",
                "input_schema": {
                    "type": "object",
                    "required": ["model_path", "selector"],
                    "properties": {
                        "model_path": {"type": "string"},
                        "selector": {"type": "string"},
                    },
                },
            },
            {
                "name": "apply_bim_revision",
                "description": (
                    "Aplica operacoes ao JSON e gera relatorio/PNGs, sem IFC."
                ),
                "input_schema": {
                    "type": "object",
                    "required": ["model_path", "revision"],
                    "properties": {
                        "model_path": {"type": "string"},
                        "revision": {
                            "type": "object",
                            "required": ["operations"],
                        },
                        "output_dir": nullable_string,
                        "selected": nullable_selection,
                    },
                },
            },
            {
                "name": "render_bim_revision",
                "description": "Renderiza PNGs sem alterar JSON ou gerar IFC.",
                "input_schema": {
                    "type": "object",
                    "required": ["model_path"],
                    "properties": {
                        "model_path": {"type": "string"},
                        "output_dir": nullable_string,
                        "selected": nullable_selection,
                    },
                },
            },
            {
                "name": "export_approved_bim_revision",
                "description": "Gera IFC somente apos aprovacao humana explicita.",
                "input_schema": {
                    "type": "object",
                    "required": ["model_path", "output_ifc", "approved"],
                    "properties": {
                        "model_path": {"type": "string"},
                        "output_ifc": {"type": "string"},
                        "approved": {"const": True},
                        "config_path": nullable_string,
                        "config": {"type": ["object", "null"]},
                        "overwrite": {"type": "boolean"},
                    },
                },
            },
        ],
        "operations": SUPPORTED_OPERATIONS,
        "language_model_required": False,
        "one_off_code_allowed": False,
    }
