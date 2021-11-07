
## Calculate ShapeNet classes mean sizes
```sh
python -m shapenet.config --output-name mean_sizes
```

## Create scenes
```sh
python -m shapenet.create --mode train --output-name dev --min-nb-objects 8 --max-nb-objects 16 --pc-nb-samples 1024 --nb-scenes 10000
python -m shapenet.create --mode val --output-name dev --min-nb-objects 8 --max-nb-objects 16 --pc-nb-samples 1024 --nb-scenes 100
```