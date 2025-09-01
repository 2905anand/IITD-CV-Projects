# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -I/opt/homebrew/Cellar/gmp/6.3.0/include -L/opt/homebrew/Cellar/gmp/6.3.0/lib -lgmp -lgmpxx

# Target executable name
TARGET = objectCode

# Source files
SRC = main.cpp

# Rule for compiling the program
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Rule to clean up compiled files
clean:
	rm -f $(TARGET)

# Rule to run the program with a specified number of processes
run: $(TARGET)
	./$(TARGET)
	









# # Compiler
# CXX = g++ --std=c++20
# CXXFLAGS = -I/opt/homebrew/Cellar/gmp/6.3.0/include -L/opt/homebrew/Cellar/gmp/6.3.0/lib -lgmpxx -lgmp

# # Target executable and source file
# TARGET = main
# SRC = main.cpp

# # Default rule to build the program
# all: $(TARGET)

# # Build the target
# $(TARGET): $(SRC)
# 	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS)

# # Clean the directory
# clean:
# 	rm -f $(TARGET)

# # Run the program
# run: all
# 	./$(TARGET)




# # Compiler and flags
# CXX = mpic++
# CXXFLAGS = -std=c++17 -I/opt/homebrew/Cellar/gmp/6.3.0/include -L/opt/homebrew/Cellar/gmp/6.3.0/lib -lgmp -lgmpxx

# # Target executable name
# TARGET = make

# # Source files
# SRC = main.cpp

# # Rule for compiling the program
# all: $(TARGET)

# $(TARGET): $(SRC)
# 	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# # Rule to clean up compiled files
# clean:
# 	rm -f $(TARGET)

# # Rule to run the program
# run: $(TARGET)
# 	./$(TARGET)
