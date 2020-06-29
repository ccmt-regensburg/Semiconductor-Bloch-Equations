#!bin/bash/

numbers=("100" "300" "1000" "3000")
gauges=("length" "velocity")
#numbers=("100")
transient="True"

for gauge in "${gauges[@]}"
do
	for num in "${numbers[@]}"
	do
		echo $gauge
		echo $num
		sed s/NK/$num/g template_params.txt > params.py
		sed -i s/GAUGE/$gauge/g params.py
		sed -i s/TRANSIENT/$transient/g params.py
		python do_plots.py
		#nohup python3 SBE.py > nohup.out &
		#sleep 5
	done
done
