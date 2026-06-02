close all
clear all 
clc
%%

% This script approximates the solution of the second-order ODE
%
%     x''(t) + 2*x'(t) + x(t) = sin(t),     x(0) = 0, x'(0) = 0
%
% with a polynomial written in scaled Taylor form:
%
%     P(t) = c0 + c1*t/1! + c2*t^2/2! + ... + cm*t^m/m!
%
% The factorial scaling is useful because derivatives are obtained by
% shifting the coefficient index.

% Main parameters:
% n = order of the differential equation. For this script n = 2, so two
%     initial conditions are fixed: P(0) = c0 and P'(0) = c1.
% m = degree of the polynomial approximation. Here m = 15, so the model uses
%     coefficients c0, c1, ..., c15.
% M = number of random collocation points. The ODE is sampled at these
%     points, creating an overdetermined linear system solved by least squares.
% In this script, n is mainly a readable reminder of the ODE order; the
% actual matrix sizes are written explicitly using m.

n = 2;
m = 15;
M = 10000;

% Random collocation points where the differential equation is sampled.
tk = 6.1*rand(M,1)-0.05;

% Right-hand side of the ODE.
uk0 = sin(tk);
%% 

% Initial conditions fix the first two polynomial coefficients:
% c0 = P(0), c1 = P'(0).
c0 = 0; 
c1 = 0;

% Move the known contributions of c0 and c1 to the right-hand side.
uk = uk0 - 1*(c0 + c1*tk) - 2*(c1)  ;

% T1 represents the second derivative part P''(t), using c2...cm.
T1 = tk.^[0:m-2]./factorial(0:m-2);

% T2 represents the first derivative part P'(t), using c2...cm.
T2 = tk.^[1:m-1]./factorial(1:m-1);

% T3 represents the polynomial part P(t), excluding fixed c0 and c1 terms.
T3 = tk.^[2:m]./factorial(2:m);

% Linear system for the unknown coefficients:
%
%     P''(tk) + 2*P'(tk) + P(tk) = sin(tk)
%
% at all sampled collocation points tk.
A = T1 + 2*T2 + 1*T3;

% Solve the overdetermined linear system in the least-squares sense.
ch = pinv(A)*uk;

% Full coefficient vector, including the two fixed initial-condition terms.
c = [c0;c1;ch];

%%

% Dense grid used only for plotting.
t = linspace(0,6,10000);

% Evaluate the polynomial model P(t).
P = 0;
for i = 0:m
    P = P + c(i+1)*t.^i/factorial(i);
end

% Exact solution of the second-order ODE with the chosen initial conditions.
x = ( exp(-t) + t.*exp(-t) - cos(t))/2;

% Print coefficients and the RMSE (root mean squared error) on the plotting
% grid. RMSE is a single number that summarizes the average size of P(t)-x(t).
c
rmse_error = sqrt(mean((P-x).^2));
rmse_error

% Plot the polynomial approximation and the exact solution.
f1=figure('DefaultAxesFontSize',22);
subplot(2,1,1)
set(gcf, 'renderer', 'painters','Position', [100 100 1200 900]);
plot(t,P,'-','LineWidth',3,'Color',[0,0,1]), hold on, 
plot(t,x,'--','LineWidth',4,'Color',[1 0 0]),
legend('Polynomial model','Exact solution','Location','northeast')
title('Polynomial model solution and exact solution', 'FontSize', 20);
xlabel('t', 'FontSize', 20);
ylabel('P,x', 'FontSize', 20);
xlim([0 6]);
ylim([-0.5 0.65] )
grid on
subplot(2,1,2)
set(gcf, 'renderer', 'painters','Position', [100 100 1200 900]);
plot(t,P - x,'-','LineWidth',3,'Color',[0,0,1]), hold on, 
% legend('Polynomial model','Exact solution','Location','northeast')
title('Pointwise error', 'FontSize', 20);
xlabel('t', 'FontSize', 20);
ylabel('P - x', 'FontSize', 20);
xlim([0 6 ]);
% ylim([0 1] )
grid on

%%

% Evaluate the first derivative P'(t).
Pd = 0;
for i = 0:m-1
    Pd = Pd + c(i+2)*t.^i/factorial(i);
end

% Exact first derivative of the exact solution.
xd =  ( sin(t) - t.*exp(-t) )/2



% Plot the first derivative and its error.
f2=figure('DefaultAxesFontSize',22);
subplot(2,1,1)
set(gcf, 'renderer', 'painters','Position', [100 100 1200 900]);
plot(t,Pd,'-','LineWidth',3,'Color',[0,0,1]), hold on, 
plot(t,xd,'--','LineWidth',4,'Color',[1 0 0]),
legend('Polynomial model','Exact solution','Location','northeast')
title('Polynomial model solution and exact solution - derivative', 'FontSize', 20);
xlabel('t', 'FontSize', 20);
ylabel('P'',x''', 'FontSize', 20);
xlim([0 6 ]);
ylim([-0.6 0.4] )
grid on
subplot(2,1,2)
set(gcf, 'renderer', 'painters','Position', [100 100 1200 900]);
plot(t,Pd - xd,'-','LineWidth',3,'Color',[0,0,1]), hold on, 
% legend('Polynomial model','Exact solution','Location','northeast')
title('Pointwise error', 'FontSize', 20);
xlabel('t', 'FontSize', 20);
ylabel('P'' - x''', 'FontSize', 20);
xlim([0 6 ]);
% ylim([0 1] )
grid on


%%

%%

% Evaluate the second derivative P''(t).
Pdd = 0;
for i = 0:m-2
    Pdd = Pdd + c(i+3)*t.^i/factorial(i);
end

% Exact second derivative of the exact solution.
xdd =  ( t.*exp(-t) - exp(-t) + cos(t) )/2;

% Plot the second derivative and its error.
f3=figure('DefaultAxesFontSize',22);
subplot(2,1,1)
set(gcf, 'renderer', 'painters','Position', [100 100 1200 900]);
plot(t,Pdd,'-','LineWidth',3,'Color',[0,0,1]), hold on, 
plot(t,xdd,'--','LineWidth',4,'Color',[1 0 0]),
legend('Polynomial model','Exact solution','Location','southeast')
title('Polynomial model solution and exact solution - second derivative', 'FontSize', 20);
xlabel('t', 'FontSize', 20);
ylabel('P'''',x''''', 'FontSize', 20);
xlim([0 6 ]);
ylim([-0.5 0.501] )
grid on
subplot(2,1,2)
set(gcf, 'renderer', 'painters','Position', [100 100 1200 900]);
plot(t,Pdd - xdd,'-','LineWidth',3,'Color',[0,0,1]), hold on, 
% legend('Polynomial model','Exact solution','Location','northeast')
title('Pointwise error', 'FontSize', 20);
xlabel('t', 'FontSize', 20);
ylabel('P'''' - x''''', 'FontSize', 20);
xlim([0 6 ]);
% ylim([0 1] )
grid on
% 
% 
