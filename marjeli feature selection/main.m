%_________________________________________________________________________%
%  
% Hybrid Whale Optimization Algorithm 
% with Simulated Annealing for Feature Selection 
%           By: Majdi Mafarja and Seyedali Mirjalili   
%           email: mmafarjeh@gmail.com
% 
% Main paper: M. Mafarja and S. Mirjalili                                 %
%               Hybrid Whale Optimization Algorithm                       %
%               with Simulated Annealing for Feature Selection            %
%               Neurocomputing , in press,                                %
%               DOI: https://doi.org/10.1016/j.neucom.2017.04.053         %
%                                                                         %
%  Developed in MATLAB R2014a                                             %
%                                                                         %
%  the original code of WOA is availble on                                %
%                                                                         %
%       Homepage: http://www.alimirjalili.com                             %
%                e-Mail: ali.mirjalili@gmail.com                          %
%                      
%_________________________________________________________________________%

clear all
clc
global A trn vald a;
SearchAgents_no=5; % Number of search agents
Max_iteration=100; % Maximum numbef of iterations

A=load('zoo.dat');
sumScore = 0;
minScore = 1;
maxScore = 0;
r=randperm(size(A,1));
trn=r(1:floor(length(r)/2));
vald=r(floor(length(r)/2)+1:end);
tic
[Best_score,Best_pos,WOA_cg_curve,fName]=WOASA(SearchAgents_no,(Max_iteration),0,1,size(A,2)-1,'AccSz');
time = toc;
acc = Acc(Best_pos);
fprintf('Acc  %f\tFitness:  %f\tSolution:  %s\tDimention: %d\tTime:  %f\n',acc,Best_score,num2str(Best_pos,'%1d'),sum(Best_pos(:)),time);

