seed=1
sh ./run_cmd.sh ${seed} 010 naive
sh ./run_cmd.sh ${seed} 020 naive
sh ./run_cmd.sh ${seed} 030 naive
sh ./run_cmd.sh ${seed} 010 dissipative
sh ./run_cmd.sh ${seed} 020 dissipative
sh ./run_cmd.sh ${seed} 030 dissipative
sh ./run_cmd.sh ${seed} 010 l2stable
sh ./run_cmd.sh ${seed} 020 l2stable
sh ./run_cmd.sh ${seed} 030 l2stable
sh ./run_cmd.sh ${seed} 010 stable
sh ./run_cmd.sh ${seed} 020 stable
sh ./run_cmd.sh ${seed} 030 stable

