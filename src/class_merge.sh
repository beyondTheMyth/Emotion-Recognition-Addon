#!/usr/bin/env bash
#kane cd sto fakelo tou database ston terminal
#to scriptaki tha antikatastisei tous fakelous me onomata sto String emotions, me ena neo fakelo me onoma sto String var
emotions="angry disgust fear sad"
var="negative"
base=$PWD;
for emo in $emotions; do
	find . -type d -name $emo > $emo.txt
	cat $emo.txt | while read line; do
		cd $line
		mkdir -p ../$var
		mv * ../$var
		cd ..
		rmdir $emo
		cd $base
	done
	rm $emo.txt
done
