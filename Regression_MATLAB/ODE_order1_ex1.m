close all
clear all 
clc

%%

% This script approximates the solution of the first-order ODE
%
%     x'(t) + 2*x(t) = 1,     x(0) = 1
%
% with a polynomial written in scaled Taylor form:
%
%     P(t) = c0 + c1*t/1! + c2*t^2/2! + ... + cm*t^m/m!
%
% The scaled basis makes derivatives simple: differentiating shifts the
% coefficient index instead of creating extra powers and factors.

% Main parameters:
% n = order of the differential equation. For this script n = 1, so one
%     initial condition is fixed: P(0) = c0.
% m = degree of the polynomial approximation. Here m = 15, so the model uses
%     coefficients c0, c1, ..., c15.
% M = number of random collocation points. The ODE is sampled at these
%     points, creating an overdetermined linear system solved by least squares.
% In this script, n is mainly a readable reminder of the ODE order; the
% actual matrix sizes are written explicitly using m.

n = 1;
m = 15;
M = 1000;

% Random collocation points. The ODE residual is forced to be small at
% these points in a least-squares sense.
tk = 4.1*rand(M,1)-0.05;

% Right-hand side of the ODE: x' + 2*x = 1.
uk0 = 1*ones(size(tk));
%% 

% The initial condition x(0) = 1 fixes the first polynomial coefficient.
% The remaining coefficients are computed from the ODE.
t0 = 1;
c0 = t0;

% Move the known contribution of c0 to the right-hand side.
uk = uk0 - 2*c0;

% T1 represents the derivative part P'(t), using coefficients c1...cm.
T1 = tk.^[0:m-1]./factorial(0:m-1);

% T2 represents the polynomial part P(t), without the fixed c0 term.
T2 = tk.^[1:m]./factorial(1:m);

% Linear system for the unknown coefficients:
%
%     P'(tk) + 2*P(tk) = 1
%
% at all sampled collocation points tk.
A = T1 + 2*T2;

% Solve the overdetermined linear system in the least-squares sense.
ch = pinv(A)*uk;

% Full coefficient vector, including the fixed initial-condition term c0.
c = [c0;ch];

%%

% Dense grid used only for plotting and measuring the error.
t = linspace(0,4,10000);

% Evaluate the polynomial model P(t).
P = 0;
for i = 0:m
    P = P + c(i+1)*t.^i/factorial(i);
end

% Exact solution of x' + 2*x = 1, x(0) = 1.
x = 0.5*(1+exp(-2*t));

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
xlim([0 4]);
ylim([-0.1 1.1] )
grid on
subplot(2,1,2)
set(gcf, 'renderer', 'painters','Position', [100 100 1200 900]);
plot(t,P - x,'-','LineWidth',3,'Color',[0,0,1]), hold on, 
% legend('Polynomial model','Exact solution','Location','northeast')
title('Pointwise error', 'FontSize', 20);
xlabel('t', 'FontSize', 20);
ylabel('P - x', 'FontSize', 20);
xlim([0 4 ]);
% ylim([0 1] )
grid on
%%

% Evaluate the derivative P'(t) from the same coefficient vector.
Pd = 0;
for i = 0:m-1
    Pd = Pd + c(i+2)*t.^i/factorial(i);
end

% Exact derivative of the exact solution.
xd = -exp(-2*t);



% Plot the derivative and its error.
f2=figure('DefaultAxesFontSize',22);
subplot(2,1,1)
set(gcf, 'renderer', 'painters','Position', [100 100 1200 900]);
plot(t,Pd,'-','LineWidth',3,'Color',[0,0,1]), hold on, 
plot(t,xd,'--','LineWidth',4,'Color',[1 0 0]),
legend('Polynomial model','Exact solution','Location','southeast')
title('Polynomial model solution and exact solution - derivative', 'FontSize', 20);
xlabel('t', 'FontSize', 20);
ylabel('P'',x''', 'FontSize', 20);
xlim([0 4 ]);
ylim([-1.1 0.1] )
grid on
subplot(2,1,2)
set(gcf, 'renderer', 'painters','Position', [100 100 1200 900]);
plot(t,Pd - xd,'-','LineWidth',3,'Color',[0,0,1]), hold on, 
% legend('Polynomial model','Exact solution','Location','northeast')
title('Pointwise error', 'FontSize', 20);
xlabel('t', 'FontSize', 20);
ylabel('P'' - x''', 'FontSize', 20);
xlim([0 4 ]);
% ylim([0 1] )
grid on

