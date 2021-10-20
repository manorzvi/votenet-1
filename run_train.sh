
x=_shapenet_num_target1_cond_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --max_epoch 40 --num_target 1
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --max_epoch 40 --num_target 1
echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes  --num_target 1 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes  --num_target 1 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

x=${x}_neg_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --max_epoch 40 --num_target 1
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --max_epoch 40 --num_target 1
echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --num_target 1 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --num_target 1 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

x=_shapenet_w_two_backbones_num_target1_cond_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_two_backbones --max_epoch 40 --num_target 1
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_two_backbones --max_epoch 40 --num_target 1
     echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_two_backbones --num_target 1 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
          python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_two_backbones --num_target 1 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

x=${x}_neg_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --use_two_backbones --max_epoch 40 --num_target 1
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --use_two_backbones --max_epoch 40 --num_target 1
echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --use_two_backbones --num_target 1 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --use_two_backbones --num_target 1 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

echo '------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

x=_shapenet_num_target3_cond_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --max_epoch 40 --num_target 3
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --max_epoch 40 --num_target 3
echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes  --num_target 3 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes  --num_target 3 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

x=${x}_neg_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --max_epoch 40 --num_target 3
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --max_epoch 40 --num_target 3
echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --num_target 3 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --neg_votes_factor 2.0 --num_target 3 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

x=_shapenet_w_two_backbones_num_target3_cond_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_two_backbones --max_epoch 40 --num_target 3
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_two_backbones --max_epoch 40 --num_target 3
     echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_two_backbones --num_target 3 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
          python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_two_backbones --num_target 3 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

x=${x}_neg_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --use_two_backbones --max_epoch 40 --num_target 3
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir log${x} --dump_results --dump_dir dump${x} --use_cond_votes --use_neg_votes --use_two_backbones --max_epoch 40 --num_target 3
echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --use_two_backbones --num_target 3 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path log${x}/checkpoint.tar --dump_dir eval${x} --use_cond_votes --use_neg_votes --use_two_backbones --num_target 3 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
