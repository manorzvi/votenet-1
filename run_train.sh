
log_prefix=log_shapenet
dump_prefix=dump_shapenet
eval_prefix=eval_shapenet
suffix=''

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir $log_prefix --dump_results --dump_dir $dump_prefix
CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir $log_prefix --dump_results --dump_dir $dump_prefix

echo python eval.py  --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}/checkpoint.tar --dump_dir $eval_prefix --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
python eval.py  --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}/checkpoint.tar --dump_dir $eval_prefix --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

suffix=${suffix}_cond_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir ${log_prefix}${suffix} --dump_results --dump_dir ${dump_prefix}${suffix} --use_cond_votes
CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir ${log_prefix}${suffix} --dump_results --dump_dir ${dump_prefix}${suffix} --use_cond_votes

echo python eval.py  --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}${suffix}/checkpoint.tar --dump_dir ${eval_prefix}${suffix} --use_cond_votes --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
python eval.py  --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}${suffix}/checkpoint.tar --dump_dir ${eval_prefix}${suffix} --use_cond_votes --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

suffix=${suffix}_rand_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir ${log_prefix}${suffix} --dump_results --dump_dir ${dump_prefix}${suffix} --use_cond_votes --use_rand_votes
CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir ${log_prefix}${suffix} --dump_results --dump_dir ${dump_prefix}${suffix} --use_cond_votes --use_rand_votes

echo python eval.py  --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}${suffix}/checkpoint.tar --dump_dir ${eval_prefix}${suffix} --use_cond_votes --use_rand_votes --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
python eval.py  --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}${suffix}/checkpoint.tar --dump_dir ${eval_prefix}${suffix} --use_cond_votes --use_rand_votes --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

log_prefix=${log_prefix}_w_two_backbones
dump_prefix=${dump_prefix}_w_two_backbones
eval_prefix=${eval_prefix}_w_two_backbones
suffix=''

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir $log_prefix --dump_results --dump_dir $dump_prefix --use_two_backbones
CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir $log_prefix --dump_results --dump_dir $dump_prefix --use_two_backbones

echo python eval.py  --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}/checkpoint.tar --dump_dir ${eval_prefix} --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
python eval.py  --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}/checkpoint.tar --dump_dir ${eval_prefix} --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

suffix=${suffix}_cond_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir ${log_prefix}${suffix} --dump_results --dump_dir ${dump_prefix}${suffix} --use_cond_votes --use_two_backbones
CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir ${log_prefix}${suffix} --dump_results --dump_dir ${dump_prefix}${suffix} --use_cond_votes --use_two_backbones

echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}${suffix}/checkpoint.tar --dump_dir ${eval_prefix}${suffix} --use_cond_votes --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
python eval.py --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}${suffix}/checkpoint.tar --dump_dir ${eval_prefix}${suffix} --use_cond_votes --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

suffix=${suffix}_rand_votes

echo CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir ${log_prefix}${suffix} --dump_results --dump_dir ${dump_prefix}${suffix} --use_cond_votes --use_rand_votes --use_two_backbones
CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir ${log_prefix}${suffix} --dump_results --dump_dir ${dump_prefix}${suffix} --use_cond_votes --use_rand_votes --use_two_backbones

echo python eval.py --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}${suffix}/checkpoint.tar --dump_dir ${eval_prefix}${suffix} --use_cond_votes --use_rand_votes --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
python eval.py --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}${suffix}/checkpoint.tar --dump_dir ${eval_prefix}${suffix} --use_cond_votes --use_rand_votes --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

