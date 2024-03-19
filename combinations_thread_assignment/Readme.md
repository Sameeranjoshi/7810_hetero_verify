Various combinations of tests used in this folder.

Current set of tests : Naming convention I_<init>_P_<producer>_C_<consumer>.cu


init: cpu, producer: cpu, consumer: cpu
init: cpu, producer: cpu, consumer: gpu
init: cpu, producer: gpu, consumer: cpu
init: cpu, producer: gpu, consumer: gpu

init: gpu, producer: cpu, consumer: cpu
init: gpu, producer: cpu, consumer: gpu
init: gpu, producer: gpu, consumer: cpu
init: gpu, producer: gpu, consumer: gpu



In future see if these are interesting to try

Init: CPU, Producer: CPU, Consumer: CPU
Init: CPU, Producer: CPU, Consumer: GPU1
Init: CPU, Producer: CPU, Consumer: GPU2
Init: CPU, Producer: GPU1, Consumer: CPU
Init: CPU, Producer: GPU1, Consumer: GPU1
Init: CPU, Producer: GPU1, Consumer: GPU2
Init: CPU, Producer: GPU2, Consumer: CPU
Init: CPU, Producer: GPU2, Consumer: GPU1
Init: CPU, Producer: GPU2, Consumer: GPU2
Init: GPU1, Producer: CPU, Consumer: CPU
Init: GPU1, Producer: CPU, Consumer: GPU1
Init: GPU1, Producer: CPU, Consumer: GPU2
Init: GPU1, Producer: GPU1, Consumer: CPU
Init: GPU1, Producer: GPU1, Consumer: GPU1
Init: GPU1, Producer: GPU1, Consumer: GPU2
Init: GPU1, Producer: GPU2, Consumer: CPU
Init: GPU1, Producer: GPU2, Consumer: GPU1
Init: GPU1, Producer: GPU2, Consumer: GPU2
Init: GPU2, Producer: CPU, Consumer: CPU
Init: GPU2, Producer: CPU, Consumer: GPU1
Init: GPU2, Producer: CPU, Consumer: GPU2
Init: GPU2, Producer: GPU1, Consumer: CPU
Init: GPU2, Producer: GPU1, Consumer: GPU1
Init: GPU2, Producer: GPU1, Consumer: GPU2
Init: GPU2, Producer: GPU2, Consumer: CPU
Init: GPU2, Producer: GPU2, Consumer: GPU1
Init: GPU2, Producer: GPU2, Consumer: GPU2