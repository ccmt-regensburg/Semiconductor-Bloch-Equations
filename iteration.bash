#!bin/bash/

numbers=(100 300 1000 3000)
gauges=("length" "velocity")
numbers=(10000)
transient=1
realistic=1
plotten=0

for gauge in "${gauges[@]}"
do
	for num in "${numbers[@]}"
	do
		if [ $plotten -gt 0 ]
		then
			python3 run.py $num $gauge $transient $realistic $plotten
		else
			nohup python3 run.py $num $gauge $transient $realistic $plotten > nohup.out &
			sleep 3
		fi
	done
done
