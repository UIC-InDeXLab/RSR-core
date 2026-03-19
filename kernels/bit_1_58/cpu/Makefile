CC = gcc
CFLAGS = -O3 -Wall -march=native -fomit-frame-pointer

all: bitnet_ternary.so tmac_ternary.so rsr_ternary.so rsr_ternary_prep.so rsr_ternary_v1_5.so rsr_ternary_v1_6.so

bitnet_ternary.so: bitnet_ternary.c
	$(CC) $(CFLAGS) -mavx2 -shared -fPIC -o bitnet_ternary.so bitnet_ternary.c -lm

tmac_ternary.so: tmac_ternary.c
	$(CC) $(CFLAGS) -mavx2 -shared -fPIC -o tmac_ternary.so tmac_ternary.c -lm

rsr_ternary.so: rsr_ternary.c
	$(CC) $(CFLAGS) -mavx2 -fopenmp -shared -fPIC -o rsr_ternary.so rsr_ternary.c

rsr_ternary_prep.so: rsr_ternary_prep.c
	$(CC) $(CFLAGS) -fopenmp -shared -fPIC -o rsr_ternary_prep.so rsr_ternary_prep.c

rsr_ternary_v1_5.so: rsr_ternary_v1_5.c
	$(CC) $(CFLAGS) -mavx2 -fopenmp -shared -fPIC -o rsr_ternary_v1_5.so rsr_ternary_v1_5.c

rsr_ternary_v1_6.so: rsr_ternary_v1_6.c
	$(CC) $(CFLAGS) -mavx2 -fopenmp -shared -fPIC -o rsr_ternary_v1_6.so rsr_ternary_v1_6.c

clean:
	rm -f *.so
