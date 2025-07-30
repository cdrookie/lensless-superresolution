% MATLAB mass spring damper simulation, including a simple backstepping controller

% I used ode solver ode45, you may try different ones and check the results
clc
close all
clear
set(0,'defaultAxesFontName', 'Times New Roman');

plot_font_size  = 40;
save_pictures   = 1;

color1 = [0, 0.4470, 0.7410];
color2 = [0.8500, 0.3250, 0.0980];
color3 = [0.9290, 0.6940, 0.1250];
color4 = [0.4940, 0.1840, 0.5560];
color5 = [0.4660, 0.6740, 0.1880];
color6 = [0.3010, 0.7450, 0.9330];
color7 = [0.6350, 0.0780, 0.1840];

System parameters
m = 1; % Units in kg
b = 0.5; % Units in Ns/m (newton times second per meter)
k1 = 2; % Units in N/m
k2 = 3; % Units in N/m

A = [0 1 0;
    -(k1+k2)/m 0 k1/m;
    k1/b 0 -k1/b];

B = [0;
    1/m;
    0];

C = [0 0 1];

Simulation parameters
% Span of simulation: [t0 tf]
tspan = [0 20];

% Solver options
opts = odeset('RelTol',1e-2,'AbsTol',1e-4,'MaxStep',0.01);

% Initial condition of the system
x0 = [1;  % Initial x1 displacement
    0;  % Initial speed associated with x1
    0]; % Initial x2 displacement

[t,x] = ode45(@(t,x) msd_solver(t,x,A,B,m,k1,k2), tspan, x0, opts);

Plot Results
f = figure('name', 'Mass Spring Damper Results','units', 'normalized','outerposition', [0 0 1 1]);

set(gca,'fontsize', plot_font_size, 'FontWeight', 'bold');
hold on
grid on

plot(t, x(:,1),'LineWidth',4,'Color',color1);
plot(t, x(:,2),'LineWidth',4,'Color',color2);
plot(t, x(:,3),'LineWidth',4,'Color',color3);
plot(t, 0.2*ones(length(t),1),'--','LineWidth',4,'Color',color4);

title(sprintf('$k_1 = %.2f$  $k_2 = %.2f$  $m = %.2f$  $b = %.2f$',...
    k1, k2, m, b),...
    'Interpreter','latex');
xlabel('time $t$ (seconds)', 'Interpreter','latex')

legend( '$x_1$ (m)',...
    '$\dot{x}_1$ (m/s)',...
    '$x_2$ (m)',...
    '$x_{1,\rm{ref}}$ (m)',...
    'Interpreter', 'latex',...
    'Location','northeast',...
    'FontSize', 80)

if save_pictures == 1
    print -dpng mass_spring_damper_control_example
end

My solver function
function dxdt = msd_solver(~,x,A,B,m,k1,k2)

% Static reference
x1_ref      = 0.2;
x1_ref_dot  = 0;
x1_ref_ddot  = 0;

c1 = 1;

f = m*(...
    - (x(1) - x1_ref)...
    + (k1 + k2)/m*x(1)...
    - k1/m*x(3)...
    + x1_ref_ddot)...
    -c1*(x(2) - x1_ref_dot);

u = f;

dxdt = A*x + B*u;

end

