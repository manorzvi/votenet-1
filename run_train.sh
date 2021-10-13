
log_prefix=log_shapenet
dump_prefix=dump_shapenet
eval_prefix=eval_shapenet
suffix=''

CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir $log_prefix --dump_results --dump_dir $dump_prefix --no_height
python eval.py  --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}/checkpoint.tar --dump_dir $eval_prefix --no_height --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

suffix=${suffix}_cond_votes

CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir ${log_prefix}${suffix} --dump_results --dump_dir ${dump_prefix}${suffix} --no_height --use_cond_votes
python eval.py  --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}${suffix}/checkpoint.tar --dump_dir ${eval_prefix}${suffix} --no_height --use_cond_votes --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

suffix=${suffix}_rand_votes

CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir ${log_prefix}${suffix} --dump_results --dump_dir ${dump_prefix}${suffix} --no_height --use_cond_votes --use_rand_votes
python eval.py  --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}${suffix}/checkpoint.tar --dump_dir ${eval_prefix}${suffix} --no_height --use_cond_votes --use_rand_votes --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

log_shapenet=${log_shapenet}_w_two_backbones
dump_shapenet=${dump_shapenet}_w_two_backbones
eval_shapenet=${eval_shapenet}_w_two_backbones
suffix=''

CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir $log_prefix --dump_results --dump_dir $dump_prefix --no_height --use_two_backbones
python eval.py  --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}/checkpoint.tar --dump_dir ${eval_prefix} --no_height --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

suffix=${suffix}_cond_votes

CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir ${log_prefix}${suffix} --dump_results --dump_dir ${dump_prefix}${suffix} --no_height --use_cond_votes --use_two_backbones
python eval.py --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}${suffix}/checkpoint.tar --dump_dir ${eval_prefix}${suffix} --no_height --use_cond_votes --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

suffix=${suffix}_rand_votes

CUDA_VISIBLE_DEVICES=0 python train.py --dataset shapenet --model cond_votenet --log_dir ${log_prefix}${suffix} --dump_results --dump_dir ${dump_prefix}${suffix} --no_height --use_cond_votes --use_rand_votes --use_two_backbones
python eval.py --dataset shapenet --model cond_votenet --checkpoint_path ${log_prefix}${suffix}/checkpoint.tar --dump_dir ${eval_prefix}${suffix} --no_height --use_cond_votes --use_rand_votes --use_two_backbones --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

