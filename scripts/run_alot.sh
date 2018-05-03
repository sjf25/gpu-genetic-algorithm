for i in {1..5}; do
	$@
done | grep -P -o "(?<=elapsed time: ).*" | awk '{total += $1; count++}END {print total/count}'
