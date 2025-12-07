@echo off
REM Build script for Windows

echo Building Lament Engine...

REM Create build directory
if not exist build mkdir build

REM Compile C files
gcc -O3 -Wall -Wextra -std=c99 -pthread -c src\model.c -o build\model.o
gcc -O3 -Wall -Wextra -std=c99 -pthread -c src\tokenizer.c -o build\tokenizer.o
gcc -O3 -Wall -Wextra -std=c99 -pthread -c src\matops.c -o build\matops.o
gcc -O3 -Wall -Wextra -std=c99 -pthread -c src\io.c -o build\io.o
gcc -O3 -Wall -Wextra -std=c99 -pthread -c src\safety.c -o build\safety.o
gcc -O3 -Wall -Wextra -std=c99 -pthread -c src\main.c -o build\main.o

REM Link
gcc build\*.o -o lament.exe -lm -pthread

if exist lament.exe (
    echo Build successful: lament.exe
) else (
    echo Build failed!
    exit /b 1
)

