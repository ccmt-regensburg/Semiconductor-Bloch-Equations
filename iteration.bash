#!bin/bash/

numbers=("10" "30" "100" "300" "1000" "3000")
gauges=("length" "velocity")
#numbers=("10" "30")

for gauge in "${gauges[@]}"
do
	for num in "${numbers[@]}"
	do
		echo $gauge
		echo $num
		sed s/NK/$num/g template_params.txt > params.py
		sed -i s/GAUGE/$gauge/g params.py
		python do_plots.py
		#nohup python3 SBE.py > nohup.out &
		#sleep 3
	done
done
