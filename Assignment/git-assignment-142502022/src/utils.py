def sort(arr):
	print("Your given array: ",arry)
	n = len(arry)
	for i in range(0,n):
		for j in range(0,n-i-1):
			if arry[j] >arry[j+1]:
				swap(arry[j],arry[j+1])
	return arry
