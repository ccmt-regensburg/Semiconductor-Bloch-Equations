#!bin/bash/

numbers=(100 300 1000 3000 10000)
numbers=(3000)
gauges=("length" "velocity")
mus=(-200 0 200)
mus=(-400 400)
transient=1
realistic=1
order=1					#Possible orders: 0 simulation, 1 plotten, 2 calculation of polarization rotation	

for gauge in "${gauges[@]}"
do
	for num in "${numbers[@]}"
	do
		for mu in "${mus[@]}"
		do
			if [ $order -gt 0 ]
			then
				python3 run.py $num $gauge $mu $transient $realistic $order
			else
				nohup python3 run.py $num $gauge $mu $transient $realistic $order > nohup.out &
				#python3 run.py $num $gauge $mu $transient $realistic $order
				sleep 3
			fi
		done
	done
done
