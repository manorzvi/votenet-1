
x=_shapenet_num_target4_cond_votes
#x=${x}_neg_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir results-oct27/log${x} --dump_results --dump_dir results-oct27/dump${x} --use_cond_votes --max_epoch 120 --num_target 4
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir results-oct27/log${x} --dump_results --dump_dir results-oct27/dump${x} --use_cond_votes --max_epoch 120 --num_target 4
echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path results-oct27/log${x}/checkpoint.tar --dump_dir results-oct27/eval${x} --use_cond_votes --num_target 4 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path results-oct27/log${x}/checkpoint.tar --dump_dir results-oct27/eval${x} --use_cond_votes --num_target 4 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

echo '------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

x=_shapenet_num_target32_cond_votes
#x=${x}_neg_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir results-oct27/log${x} --dump_results --dump_dir results-oct27/dump${x} --use_cond_votes --max_epoch 120 --num_target 32
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir results-oct27/log${x} --dump_results --dump_dir results-oct27/dump${x} --use_cond_votes --max_epoch 120 --num_target 32
echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path results-oct27/log${x}/checkpoint.tar --dump_dir results-oct27/eval${x} --use_cond_votes --num_target 32 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path results-oct27/log${x}/checkpoint.tar --dump_dir results-oct27/eval${x} --use_cond_votes --num_target 32 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

echo '------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

x=_shapenet_num_target256_cond_votes
#x=${x}_neg_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir results-oct27/log${x} --dump_results --dump_dir results-oct27/dump${x} --use_cond_votes --max_epoch 120 --num_target 256
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir results-oct27/log${x} --dump_results --dump_dir results-oct27/dump${x} --use_cond_votes --max_epoch 120 --num_target 256
echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path results-oct27/log${x}/checkpoint.tar --dump_dir results-oct27/eval${x} --use_cond_votes --num_target 256 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path results-oct27/log${x}/checkpoint.tar --dump_dir results-oct27/eval${x} --use_cond_votes --num_target 256 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

echo '------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

x=_shapenet_num_target64_cond_votes
#x=${x}_neg_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir results-oct27/log${x} --dump_results --dump_dir results-oct27/dump${x} --use_cond_votes --max_epoch 120 --num_target 64
     CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir results-oct27/log${x} --dump_results --dump_dir results-oct27/dump${x} --use_cond_votes --max_epoch 120 --num_target 64
echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path results-oct27/log${x}/checkpoint.tar --dump_dir results-oct27/eval${x} --use_cond_votes --num_target 64 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
     python eval.py --dataset shapenet --model cond_votenet --checkpoint_path results-oct27/log${x}/checkpoint.tar --dump_dir results-oct27/eval${x} --use_cond_votes --num_target 64 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
