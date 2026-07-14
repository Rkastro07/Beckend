"""Converte ASC ASCII (xyz [rgb] [labels...]) → PLY binário com SUBSAMPLE.

Lê direto do ZIP (não extrai os 25GB). Pega 1 a cada N linhas pra reduzir
densidade. Resultado: PLY binário gerenciavel (50-200 MB tipico).

Uso:
  python asc_to_ply_subsample.py <input.zip ou .asc> [output.ply] [subsample_rate]

  subsample_rate: pega 1 ponto a cada N (default=20 → reduz 20x)
"""
import sys
import time
import zipfile
import struct
from pathlib import Path
from io import BufferedReader


def detect_columns(first_line: str) -> dict:
    """Detecta colunas do ASC pela primeira linha."""
    parts = first_line.strip().split()
    n = len(parts)
    fmt = {"n_cols": n, "has_rgb": False, "has_label": False}
    if n == 3:
        fmt["layout"] = "xyz"
    elif n == 6:
        # Pode ser xyz+rgb OU xyz+normal — assumir RGB se valores 0-255
        try:
            vals = [float(p) for p in parts[3:6]]
            if all(0 <= v <= 255 for v in vals) and any(v > 1 for v in vals):
                fmt["layout"] = "xyz_rgb"
                fmt["has_rgb"] = True
            else:
                fmt["layout"] = "xyz_nrm"
        except ValueError:
            fmt["layout"] = "xyz_extra"
    elif n == 7:
        fmt["layout"] = "xyz_rgb_label"
        fmt["has_rgb"] = True
        fmt["has_label"] = True
    elif n >= 4:
        # xyz + 1+ extra: assume última como label
        fmt["layout"] = "xyz_rgb_label" if n == 7 else f"xyz_extra_{n-3}"
        fmt["has_rgb"] = (n >= 6)
        fmt["has_label"] = True
    else:
        fmt["layout"] = "unknown"
    return fmt


def write_ply_header(fout, n_points, has_rgb, has_label):
    """Escreve header binário PLY."""
    h = ["ply", "format binary_little_endian 1.0",
         f"element vertex {n_points}",
         "property float x", "property float y", "property float z"]
    if has_rgb:
        h += ["property uchar red", "property uchar green", "property uchar blue"]
    if has_label:
        h += ["property int label"]
    h += ["end_header"]
    for line in h:
        fout.write((line + "\n").encode("ascii"))


def stream_convert(input_stream, output_path, subsample_rate=20, max_pts=None):
    """Stream conversion ASC → PLY binário com subsample."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📐 Detectando formato (lendo primeiras linhas)...")
    # Skip linhas vazias / comentários
    first_line = None
    while True:
        raw = input_stream.readline()
        if not raw:
            sys.exit("Stream vazio")
        line = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
        line = line.strip()
        if line and not line.startswith("#"):
            first_line = line
            break

    fmt = detect_columns(first_line)
    print(f"   Colunas: {fmt['n_cols']} | Layout: {fmt['layout']}")
    print(f"   RGB: {fmt['has_rgb']} | Label: {fmt['has_label']}")

    has_rgb = fmt["has_rgb"]
    has_label = fmt["has_label"]

    # Struct format
    struct_fmt = "<fff"  # xyz float
    if has_rgb:
        struct_fmt += "BBB"  # rgb uchar
    if has_label:
        struct_fmt += "i"    # label int

    bytes_per_pt = struct.calcsize(struct_fmt)
    print(f"   Bytes/pt: {bytes_per_pt}")
    print(f"   Subsample rate: 1 a cada {subsample_rate}")

    # Header com placeholder de count (vai reescrever ao final)
    with open(out_path, "wb") as fout:
        # Placeholder header — ajustar count depois
        write_ply_header(fout, 0, has_rgb, has_label)
        header_end = fout.tell()

        # Processa primeira linha
        n_written = 0
        n_read = 0

        def parse_and_write(line: str):
            parts = line.strip().split()
            if len(parts) < 3:
                return False
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                vals = [x, y, z]
                if has_rgb and len(parts) >= 6:
                    r, g, b = int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))
                    r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
                    vals += [r, g, b]
                elif has_rgb:
                    vals += [128, 128, 128]
                if has_label and len(parts) >= 7:
                    vals.append(int(float(parts[6])))
                elif has_label:
                    vals.append(0)
                fout.write(struct.pack(struct_fmt, *vals))
                return True
            except (ValueError, struct.error):
                return False

        if parse_and_write(first_line):
            n_written += 1
        n_read = 1

        # Stream rest with subsample
        t0 = time.time()
        last_report = t0
        for raw in input_stream:
            n_read += 1
            if n_read % subsample_rate != 0:
                continue
            line = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
            if parse_and_write(line):
                n_written += 1

            if max_pts and n_written >= max_pts:
                print(f"   Atingido max_pts ({max_pts:,}), parando.")
                break

            if n_read % 2_000_000 == 0:
                now = time.time()
                if now - last_report > 5:
                    rate = n_read / (now - t0)
                    print(f"   Lidas {n_read/1e6:.1f}M, escritas {n_written/1e6:.2f}M, "
                          f"velocidade {rate/1e6:.1f}M lin/s")
                    last_report = now

        # Reescreve header com count real
        print(f"\n   Total lidas: {n_read:,}")
        print(f"   Total escritas: {n_written:,}")

    # Reescreve header com count correto (precisa substituir placeholder "0")
    # Estratégia: reabrir, ler primeiro 1KB, substituir, regravar inicio
    print(f"\n   Atualizando header com count real...")
    with open(out_path, "r+b") as f:
        # Lê header atual
        f.seek(0)
        head_bytes = f.read(2048)
        head_str = head_bytes.decode("ascii", errors="ignore")
        # Substitui "element vertex 0" por count real
        new_head = head_str.replace(f"element vertex 0", f"element vertex {n_written}", 1)
        new_head_bytes = new_head.encode("ascii")
        # O novo header pode ter tamanho diferente — se sim, precisa shift
        # Pra simplicidade, padding com espaços antes do "\n" do "element vertex"
        # Versão mais simples: assumir mesmo tamanho de header (ok se count tem mesmo n_digitos)
        # Se diferir, vai ter problema. Vamos forçar mesmo tamanho.
        old_token = b"element vertex 0\n"
        new_token_str = f"element vertex {n_written}\n"
        diff = len(new_token_str) - len(old_token)
        if diff > 0:
            # Header novo é maior — precisa reabrir e reescrever tudo
            print(f"   ⚠️  Header tamanho aumentou ({diff} bytes), reescrevendo arquivo inteiro...")
            # Lê o resto
            f.seek(header_end)
            data = f.read()
            # Reescreve
            f.seek(0)
            f.truncate()
            write_ply_header(f, n_written, has_rgb, has_label)
            f.write(data)
        else:
            # Cabe — só substitui inline
            f.seek(0)
            f.write(new_head_bytes[:len(head_bytes)])

    elapsed = time.time() - t0
    final_size_mb = out_path.stat().st_size / 1024**2
    print(f"\n✅ Concluído em {elapsed:.0f}s")
    print(f"   Saída: {out_path}")
    print(f"   Tamanho: {final_size_mb:.1f} MB ({n_written:,} pontos)")


def main():
    if len(sys.argv) < 2:
        sys.exit("Uso: python asc_to_ply_subsample.py <input.zip ou .asc> [output.ply] [subsample_rate]")

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        sys.exit(f"Nao encontrado: {input_path}")

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.parent / f"{input_path.stem}_subsampled.ply"

    subsample = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Subsample: 1 a cada {subsample}")

    if input_path.suffix.lower() == ".zip":
        print(f"\n📦 Lendo do ZIP (sem extrair)...")
        with zipfile.ZipFile(input_path, "r") as zf:
            ascs = [n for n in zf.namelist()
                    if n.lower().endswith((".asc", ".txt", ".xyz", ".csv"))]
            if not ascs:
                # Tenta primeiro arquivo
                ascs = [zf.namelist()[0]]
            asc_name = ascs[0]
            print(f"   Arquivo: {asc_name}")
            print(f"   Tamanho extraído: {zf.getinfo(asc_name).file_size / 1024**3:.2f} GB")
            with zf.open(asc_name, "r") as f:
                stream_convert(f, output_path, subsample_rate=subsample)
    else:
        with open(input_path, "rb") as f:
            stream_convert(f, output_path, subsample_rate=subsample)


if __name__ == "__main__":
    main()
