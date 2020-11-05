CXX=clang++
main.exe: main.o
	$(CXX) -o $@ $^ -lOpenCL -L/opt/amdgpu-pro/lib/x86_64-linux-gnu/

%.o: %.cpp
	$(CXX) -o $@ -c $<

clean:
	rm -rf main.exe *.o
