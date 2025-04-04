from scipy.interpolate import interp1d
data = np.sin(np.arange(0,100,.001))
print(len(data))
data[10000:20000] = np.random.uniform(8,8.1,10000)
sorted_data = np.sort(data)
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data) 


x_min, x_max = 0, 9

F_min = np.interp(x_min, sorted_data, cdf)  
F_max = np.interp(x_max, sorted_data, cdf)  

mask = (sorted_data >= x_min) & (sorted_data <= x_max)
truncated_data = sorted_data[mask]
truncated_cdf = cdf[mask]

truncated_cdf = (truncated_cdf - F_min) / (F_max - F_min)

inverse_cdf_truncated = interp1d(truncated_cdf, truncated_data, kind='linear', fill_value="extrapolate")


num_samples = 100000
uniform_samples = np.random.rand(num_samples)  
sampled_values_truncated = inverse_cdf_truncated(uniform_samples)


plt.figure(figsize=(8, 5))
plt.hist(sampled_values_truncated, bins=np.arange(x_min, x_max + (max(data)-min(data))/100, (max(data)-min(data))/100), density=True, alpha=0.6, edgecolor="black", label="Sampled from Truncated CDF")


plt.hist(data, bins=np.arange(np.min(data), np.max(data)+(max(data)-min(data))/100, (max(data)-min(data))/100), density=True, alpha=0.3, edgecolor="black", label="Original Histogram")

plt.axvline(x_min, color="red", linestyle="--", label="Truncation Bounds")
plt.axvline(x_max, color="red", linestyle="--")

plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
