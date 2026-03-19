CC = gcc
CFLAGS = -O3 -Wall -march=native -fomit-frame-pointer

all: bitnet_i2s.so bitnet_official.so tmac_binary.so rsr.so rsr_v4_2.so rsr_v2_4.so rsr_prep.so

bitnet_i2s.so: bitnet_i2s.c
	$(CC) $(CFLAGS) -mavx2 -shared -fPIC -o bitnet_i2s.so bitnet_i2s.c -lm

bitnet_official.so: bitnet_official.c
	$(CC) $(CFLAGS) -mavx2 -shared -fPIC -o bitnet_official.so bitnet_official.c -lm

tmac_binary.so: tmac_binary.c
	$(CC) $(CFLAGS) -mavx2 -shared -fPIC -o tmac_binary.so tmac_binary.c -lm

rsr.so: rsr.c
	$(CC) $(CFLAGS) -mavx2 -shared -fPIC -o rsr.so rsr.c

rsr_v4_2.so: rsr_v4_2.c
	$(CC) $(CFLAGS) -mavx2 -fopenmp -shared -fPIC -o rsr_v4_2.so rsr_v4_2.c

rsr_v2_4.so: rsr_v2_4.c
	$(CC) $(CFLAGS) -mavx2 -fopenmp -shared -fPIC -o rsr_v2_4.so rsr_v2_4.c

rsr_prep.so: rsr_prep.c
	$(CC) $(CFLAGS) -fopenmp -shared -fPIC -o rsr_prep.so rsr_prep.c

clean:
	rm -f rsr.so rsr_v4_2.so bitnet_i2s.so bitnet_official.so tmac_binary.so rsr_v2_4.so rsr_prep.so
