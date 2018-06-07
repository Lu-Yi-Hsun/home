#!/bin/sh
 
macro="-I __THROW -I __THROWNL -I __nonnull -I __THROWNL -I __attribute_pure -I nonnull -I __attribute"
rm $PWD/tags
shopt -s globstar
for i in **/*
do
    if [ -f "$i" ];
    then
		base=${i##*.}

		if [ "${base}" == "c" ]; then
			gcc -M "${i}"  |\
			sed -e 's/[\\ ]/\n/g'|\
			sed -e '/^$/d' -e '/\.o:[ \t]*$/d'|\
			ctags -L -  "$macro" --file-scope=yes --langmap=c:+.h --languages=c,c++ --links=yes --c-kinds=+p --c++-kinds=+p --fields=+iaS --extra=+q --append=yes
			echo $-
		elif [ "${base}" = "cu" ]; then
			nvcc -M "${i}"|\
			sed -e 's/[\\ ]/\n/g'|\
			sed -e '/^$/d' -e '/\.o:[ \t]*$/d'|\
			ctags -L -  "$macro" --file-scope=yes --langmap=c++:+.cu --langmap=c:+.h --languages=c,c++ --links=yes --c-kinds=+p --c++-kinds=+p --fields=+iaS --extra=+q --append=yes	
		fi
    fi
done
