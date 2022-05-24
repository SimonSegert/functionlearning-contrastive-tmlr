import os
for run_id in range(3):
    for incl_y_jitter in [0,1]:
        for incl_position_jitter in [0,1]:
            for incl_rescale in [0,1]:
                #make sure at least one augmentation is present
                if max(incl_rescale,incl_position_jitter,incl_y_jitter)==0:
                    continue

                mn = 'contrastive_ablation'
                if incl_y_jitter>0:
                    mn+='_yjitter'
                if incl_position_jitter > 0:
                    mn += '_posjitter'
                if incl_rescale>0:
                    mn+='_rescale'

                args = '--batch_size 512 --n_epochs 10 --tau 0.5 --lr 0.001  --x_jitter_strength 0.4 --y_jitter_strength 0 --h_size 128 '
                if run_id==0:
                    os.mkdir(f'augmentation_ablations/{mn}')

                args+=f' --save_dir augmentation_ablations/{mn}/run{run_id} --model_name {mn} '
                args+=f' --incl_y_jitter {incl_y_jitter} --incl_position_jitter {incl_position_jitter} --incl_rescale {incl_rescale} '
                os.system('python training.py '+args)