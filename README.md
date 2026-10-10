# Inventory models

Recursively inventories SafeTensors/GGUF/checkpoint files and writes a single
self-contained HTML report (opens directly in any browser): a sortable,
groupable, live-searchable table with clickable links straight to each model
file. SafeTensors inspection reads only the header plus tiny `.comfy_quant`
metadata blobs; full model weights are not loaded into RAM.

Examples:
```
  python model_inventory.py "C:\\checkpoints"
```
```
  python model_inventory.py "C:\\checkpoints" "C:\\Lora" --output models.html
```
```
  python model_inventory.py "C:\\Models" --output models.html --json models.json
```

<img width="1588" height="1241" alt="Screen-2026-10-06_17-06-16" src="https://github.com/user-attachments/assets/e2f352cd-1810-4636-9a7d-a4cdf1ba24a6" />
