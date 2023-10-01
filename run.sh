export CUDA_VISIBLE_DEVICES=0,1
model=ada3
lr=1e-4
batch_size=50
method=cont
epochs=50
beta=0.08
txt=train_sort_A2_m,.txt
all=1

name=FaCoRNet_Adaface_lr_${lr}_beta

python train_p.py --arch ${model} --method ${method} --batch_size ${batch_size} --sample sample0 --save_path ${model}_${method}_${name}.pth --epochs ${epochs} --beta ${beta} --log_path log_${model}_${method}_${name}.txt --txt ${txt} --all ${all}

python find.py  --arch ${model} --method ${method} --sample sample0 --save_path ${model}_${method}_${name}.pth --batch_size ${batch_size} --log_path log_${model}_${method}_${name}.txt

python test.py  --arch ${model} --method ${method} --sample sample0 --save_path ${model}_${method}_${name}.pth  --batch_size ${batch_size} --log_path log_${model}_${method}_${name}.txt

