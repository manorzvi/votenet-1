# Conditional Deep Hough Voting for 3D Object Detection in Point Clouds
## Follow-up work of [Deep Hough Voting for 3D Object Detection in Point Clouds](https://github.com/facebookresearch/votenet)
Created by [Manor Zvi](mailto:manor.zvi@campus.technion.ac.il), [Nitai Fingerhut](mailto:nitaifingerhut@gmail.com)

![teaser](https://github.com/manorzvi/votenet-1/blob/shapenet-workinet-real/results-oct24/eval_shapenet_num_target32_cond_votes/000001_snap00.png)

## Table of contents

- [Installation](#installation)
- [Data Preperation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Examples](#examples)

## installation

Install [Pytorch](https://pytorch.org/get-started/locally/) and [Tensorflow](https://github.com/tensorflow/tensorflow) (for TensorBoard). It is required that you have access to GPUs. Matlab is required to prepare data for SUN RGB-D. The code is tested with Ubuntu 18.04, Pytorch v1.1, TensorFlow v1.14, CUDA 10.0 and cuDNN v7.4. Note: After a code update on 2/6/2020, the code is now also compatible with Pytorch v1.2+

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

    cd pointnet2
    python setup.py install

Install the following Python dependencies (with `pip install`):

    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'

## Data Preparation

1. Download the [Shapenet Dataset](https://shapenet.cs.stanford.edu/iccv17/)
2. Create artificial scenes: 
```shell
cd shapenet
python shape_scenes.py --step-size 1.0 --input-path train_data --output-path scenes/train --min-n-objects 4 --max-n-objects 4 --exclude bag cap earphone rocket skateboard mug --debug --rotate
```
- `--exclude` exclude one or more object classes from appearing in scenes (for non-balanced classes, for example).
- `--step-size` step size in meters according to which the scene will be created
- `--rotate` whether or not to randomly rotate objects (around the z axis)

## Train

3. Train Conditional Votenet:
```shell
CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir <log directory> --dump_results --dump_dir <dump directory> --use_cond_votes
```
- `--use_cond_votes` use conditional votes or note. In Conditional Votes mode only points on objects associated with the condition signal votes towards their respective centroid.

## Evaluation

4. Test the trained model with its checkpoint:
```shell
python eval.py --dataset shapenet --model cond_votenet --checkpoint_path <log directory>/checkpoint.tar --dump_dir {evaluation directory}/eval${x} --use_cond_votes --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
```

## Examples

### Eight Objects, Conditional Votes, large step size
![teaser](https://github.com/manorzvi/votenet-1/blob/large_step_scene.gif)
### Eight Objects, Conditional Votes, small step size
![teaser](https://github.com/manorzvi/votenet-1/blob/small_step_scene.gif)
Our method excelled to detect & classify objects, even when occluded and very crowded scenes.  
### Eight Objects, Conditional Votes, large step size, with negative votes
![teaser](https://github.com/manorzvi/votenet-1/blob/large_step_neg_votes_scene.gif)
As we see here, negative votes on un-conditioned objects, did not improve the performance.
