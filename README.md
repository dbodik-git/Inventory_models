# Inventory models

Recursively inventories SafeTensors/GGUF/checkpoint files and writes a single
self-contained HTML report (opens directly in any browser): a sortable,
groupable, live-searchable table with clickable links straight to each model
file. SafeTensors inspection reads only the header plus tiny `.comfy_quant`
metadata blobs; full model weights are not loaded into RAM.

Examples:
```
  python model_inventory.py "E:\\SD\\checkpoints"
```
```
  python model_inventory.py "E:\\SD\\checkpoints" "E:\\SD\\Lora" --output models.html
```
```
  python model_inventory.py "D:\\AI\\Models" --output models.html --json models.json
```
