import pandas as pd
import json

csv_file = "prompt_energy_analysis_table.csv"
json_file = "cuts.json"

df = pd.read_csv(csv_file,comment="#")

def parse_runs(run_field):
    if pd.isna(run_field):
        return []
    runs = []
    for r in str(run_field).split(","):
        r = r.strip()
        if r:
            runs.append(int("20" + r))
    return runs

run_lists = []
run_energies = []
cuts = []

for _, row in df.iterrows():
    energy = int(row["energy"])
    runs = parse_runs(row["run"])

    run_lists.append(runs)
    run_energies.append(energy)

    xmin, xmax = row["xmin"], row["xmax"]
    ymin, ymax = row["ymin"], row["ymax"]

    cut = (
        f"hodo_y1_single_cl_flag && "
        f"(hodo_y1_cl0_pos > {ymin} && hodo_y1_cl0_pos < {ymax}) &&"
        f"hodo_x1_single_cl_flag && hodo_x2_single_cl_flag && "
        f"(((hodo_x1_cl0_pos + hodo_x2_cl0_pos)/2) > {xmin} && "
        f"((hodo_x1_cl0_pos + hodo_x2_cl0_pos)/2) < {xmax})"
    )

    cuts.append(
        cut,
    )

out = {
    "global": {
        "run info": {
            "run list": run_lists,
            "run energies": run_energies,
            "do fitamp": True,
            "do matrix 3x3": True
        }
    },
    "cuts": cuts
}

with open(json_file, "w") as f:
    json.dump(out, f, indent=4)
