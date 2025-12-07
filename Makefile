CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -std=c99 -pthread
LDFLAGS = -lm -pthread
SRCDIR = src
OBJDIR = build
TARGET = lament

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Default target
all: $(TARGET)

# Create target binary
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Compile source files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create build directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(TARGET) $(TARGET).exe

# Install (copy to system path, optional)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Run tests
test: $(TARGET)
	@echo "Running tests..."
	@if [ -d tests ]; then \
		for test in tests/*.sh; do \
			if [ -f $$test ]; then \
				bash $$test; \
			fi \
		done \
	fi

# Debug build
debug: CFLAGS = -g -O0 -Wall -Wextra -std=c99 -pthread -DDEBUG
debug: $(TARGET)

.PHONY: all clean install test debug

