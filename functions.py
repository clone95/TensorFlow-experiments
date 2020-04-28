import numpy as np

def gen_time_series(batch_size, n_steps):
    freq1, freq2, offset1, offset2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offset1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offset2) * (freq2 * 10 + 10))
    series += 0.1 *(np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)

def gen_cnn_series(examples, lenght):
    
    series = []

    for el in range(0, examples):
        
        example = [x for x in range(el, el + lenght)]
        example.append(el + lenght + 7)
        series.append(example)

    return series

def gen_np_cnn_series(examples, lenght):
    series = np.random.rand(1, examples, 1)

    
    return series

print(gen_np_cnn_series(3,4))