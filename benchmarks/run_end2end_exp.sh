
# for gpu_name in A100-40G A100-80G
for gpu_name in A100-80G
do
    byte_per_gpu=85899345920
    if [ $gpu_name = A100-40G ]; then
        byte_per_gpu=42949672960
    fi
    # for tot_gpu_num in 8 4
    for tot_gpu_num in 8 4
    do
        gpu_ids=0,1,2,3,4,5,6,7
        if [ $tot_gpu_num -eq 4 ]; then
            gpu_ids=0,1,2,3
        fi
        for max_group_seq_num in 1 20
        do 
            top_k=20
            similar_threshold=0.2
            fully_connected_gpu_unit=2
            machine_name=zxcpu


            specify_outlen=
            # router
            for use_specify_outlen in no yes
            do
                specify_outlen=
                outlen_file_name_setting=maxlen_4096
                if [ $use_specify_outlen = yes ]; then
                    specify_outlen=--specify_outlen
                    outlen_file_name_setting=setOutlen
                fi

                echo use_specify_outlen: $use_specify_outlen  specify_outlen: $specify_outlen

                for gen_execplans_baseline in ours naive
                do 
                    if [ $max_group_seq_num -eq 20 ] && [ $gen_execplans_baseline = naive ]; then
                        continue
                    fi

                    if [ -a test_end2end_schedule/test_1231_${tot_gpu_num}gpu-router_${gen_execplans_baseline}_${gpu_name}_${machine_name}_not_multiple_choice_question_${outlen_file_name_setting}_10000_${max_group_seq_num}_1.log ]; then
                        echo skip test_1231_${tot_gpu_num}gpu-router_${gen_execplans_baseline}_${gpu_name}_${machine_name}_not_multiple_choice_question_${outlen_file_name_setting}_10000_${max_group_seq_num}_1.log
                        continue
                    fi

                    CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --test-case router --ratio-seed 0 --ratio-set 1 --reqnum 10000 --router_question_version 'not_multiple_choice_question' --max_token_num 4096  $specify_outlen --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name >> test_end2end_schedule/test_1231_${tot_gpu_num}gpu-router_${gen_execplans_baseline}_${gpu_name}_${machine_name}_not_multiple_choice_question_${outlen_file_name_setting}_10000_${max_group_seq_num}_1.log
                done
            done
            specify_outlen=
            # ensemble
            for reqnum in 1000 5000 10000 
            do
                for max_token_num in 512 256
                do
                    for gen_execplans_baseline in ours naive
                    do 

                        if [ $max_group_seq_num -eq 20 ] && [ $gen_execplans_baseline = naive ]; then
                            continue
                        fi

                        if [ -a test_end2end_schedule/test_1231_${tot_gpu_num}gpu-llm-blender_${gen_execplans_baseline}_${gpu_name}_${machine_name}_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_1.log ]; then
                            echo skip test_1231_${tot_gpu_num}gpu-llm-blender_${gen_execplans_baseline}_${gpu_name}_${machine_name}_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_1.log
                            continue
                        fi

                        CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --test-case general --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --max_token_num $max_token_num $specify_outlen --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name >> test_end2end_schedule/test_1231_${tot_gpu_num}gpu-llm-blender_${gen_execplans_baseline}_${gpu_name}_${machine_name}_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_1.log
                    done
                done
            done
        done
    done
done





