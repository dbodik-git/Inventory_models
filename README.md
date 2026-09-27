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
