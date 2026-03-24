CC = gcc
CFLAGS = -O3 -Wall -march=native -fomit-frame-pointer

all: bitnet_ternary.so tmac_ternary.so rsr_ternary.so rsr_ternary_prep.so rsr_ternary_prep_nonsquare.so rsr_ternary_v3_1.so rsr_ternary_v3_1_batch.so rsr_ternary_v3_3.so rsr_ternary_v3_3_batch.so

bitnet_ternary.so: bitnet_ternary.c
	$(CC) $(CFLAGS) -mavx2 -shared -fPIC -o bitnet_ternary.so bitnet_ternary.c -lm

tmac_ternary.so: tmac_ternary.c
	$(CC) $(CFLAGS) -mavx2 -shared -fPIC -o tmac_ternary.so tmac_ternary.c -lm

rsr_ternary.so: rsr_ternary.c
	$(CC) $(CFLAGS) -mavx2 -fopenmp -shared -fPIC -o rsr_ternary.so rsr_ternary.c

rsr_ternary_prep.so: rsr_ternary_prep.c
	$(CC) $(CFLAGS) -fopenmp -shared -fPIC -o rsr_ternary_prep.so rsr_ternary_prep.c

rsr_ternary_prep_nonsquare.so: rsr_ternary_prep_nonsquare.c
	$(CC) $(CFLAGS) -fopenmp -shared -fPIC -o rsr_ternary_prep_nonsquare.so rsr_ternary_prep_nonsquare.c

rsr_ternary_v3_1.so: rsr_ternary_v3_1.c
	$(CC) $(CFLAGS) -mavx2 -fopenmp -shared -fPIC -o rsr_ternary_v3_1.so rsr_ternary_v3_1.c

rsr_ternary_v3_1_batch.so: rsr_ternary_v3_1_batch.c
	$(CC) $(CFLAGS) -mavx2 -fopenmp -shared -fPIC -o rsr_ternary_v3_1_batch.so rsr_ternary_v3_1_batch.c -lm

rsr_ternary_v3_3.so: rsr_ternary_v3_3.c
	$(CC) $(CFLAGS) -mavx2 -fopenmp -shared -fPIC -o rsr_ternary_v3_3.so rsr_ternary_v3_3.c

rsr_ternary_v3_3_batch.so: rsr_ternary_v3_3_batch.c
	$(CC) $(CFLAGS) -mavx2 -fopenmp -shared -fPIC -o rsr_ternary_v3_3_batch.so rsr_ternary_v3_3_batch.c -lm

clean:
	rm -f *.so
