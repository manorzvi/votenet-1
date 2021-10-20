
x=_shapenet_cond_votes

#echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --max_epoch 20
#     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --max_epoch 20
#echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
#     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

#y=${x}_cond_bboxs
x=${x}_neg_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --max_epoch 80
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --max_epoch 80
echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

#echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${y} --dump_results --dump_dir dump${y} --use_cond_votes --use_cond_bboxs --max_epoch 40
#    CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${y} --dump_results --dump_dir dump${y} --use_cond_votes --use_cond_bboxs --max_epoch 40
#echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${y}/checkpoint.tar --dump_dir eval${y} --use_cond_votes --use_cond_bboxs --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
#     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${y}/checkpoint.tar --dump_dir eval${y} --use_cond_votes --use_cond_bboxs --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

x=${x}_cond_bboxs

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --use_cond_bboxs --max_epoch 80
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --use_cond_bboxs --max_epoch 80
echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --use_cond_bboxs --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --use_cond_bboxs --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

#x=_shapenet_w_two_backbones_cond_votes

#echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_two_backbones --max_epoch 20
#     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_two_backbones --max_epoch 20
#     echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
#          python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

#x=${x}_neg_votes
#y=${x}_cond_bboxs

#echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --use_two_backbones --max_epoch 20
#     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --use_two_backbones --max_epoch 20
#echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
#     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

#echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${y} --dump_results --dump_dir dump${y} --use_cond_votes --use_cond_bboxs --use_two_backbones --max_epoch 20
#     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${y} --dump_results --dump_dir dump${y} --use_cond_votes --use_cond_bboxs --use_two_backbones --max_epoch 20
#echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${y}/checkpoint.tar --dump_dir eval${y} --use_cond_votes --use_cond_bboxs --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
#     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${y}/checkpoint.tar --dump_dir eval${y} --use_cond_votes --use_cond_bboxs --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

#x=${x}_cond_bboxs

#echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --use_cond_bboxs --use_two_backbones --max_epoch 20
#     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --use_cond_bboxs --use_two_backbones --max_epoch 20
#echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --use_cond_bboxs --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
#     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --use_cond_bboxs --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
