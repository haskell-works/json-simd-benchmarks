SRC_FILES := $(wildcard *.c)
OBJ_FILES=$(patsubst %.c, %.o, $(SRC_FILES))

%.s : %.c
	gcc -O2 -mavx2 -mlzcnt -mbmi2 -S -Wall -DAVX2_ENABLED $<

%.o : %.c
	gcc -O2 -mavx2 -mlzcnt -mbmi2 -c -Wall -DAVX2_ENABLED $<

a.out : $(OBJ_FILES)
	gcc -o a.out $(OBJ_FILES)

all: $(patsubst %.c,%.s,$(SRC_FILES)) $(patsubst %.c,%.out,$(SRC_FILES))

test:
	if [ -f test.sh ]; then ./test.sh; fi

clean:
	rm -f *.s a.out *.o
