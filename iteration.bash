#!bin/bash/

numbers=(100 300 1000 3000 10000)
numbers=(400)
gauges=("length")
transient=2				#Posible values: -1 no transient, {0, 1, 2} different forms of transients
realistic=1
order=0					#Possible orders: 0 simulation, 1 plotten, 2 calculation of polarization rotation	

tra_fac=.05
nir_fac=.05

mu_min=-900
mu_max=900

mu_step=25
index=1

number_of_cores=6
run_time=$((50*60))

for gauge in "${gauges[@]}"
do
	for num in "${numbers[@]}"
	do
		if [ $order -gt 0 ]
		then
			for ((mu=mu_min;mu<=mu_max;mu+=mu_step));
			do
				python3 run.py $num $gauge $mu $transient $realistic $order $tra_fac $nir_fac 
			done
		else
			for ((mu=mu_min;mu<=mu_max;mu+=mu_step));
			do
				nohup python3 run.py $num $gauge $mu $transient $realistic $order $tra_fac $nir_fac > nohup.out &
				sleep 3

				if [ $((index%number_of_cores)) -eq 0 ] && [ $mu -lt $mu_max ]
				then
					echo "Zeit für eine Pause"
					sleep $run_time
				fi

				index=$(( index + 1))
			done
		fi
	done
done
