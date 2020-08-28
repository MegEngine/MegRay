#! /bin/bash

dirs="include src test"

for dir in $dirs; do
    pushd "${dir}" &>/dev/null
    find . \
         \( -name '*.c' \
         -o -name '*.cc' \
         -o -name '*.cpp' \
         -o -name '*.h' \
         -o -name '*.hh' \
         -o -name '*.hpp' \) \
         -exec clang-format -i '{}' \;
    popd &>/dev/null
done
