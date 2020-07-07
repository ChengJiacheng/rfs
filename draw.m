clear; close all
load('result.mat')


D1 = pdist2(features, features, 'cosine')

figure()

temp = [];
for i = 0:4
temp = [temp, D1(find(labels==i), find(labels==i))];
end
temp = temp(:);
temp(temp<1e-3) = [];

histogram(temp(:), 50, 'Normalization','probability')

% histfit(temp(:))

hold on

% temp = D1(find(labels==1), find(labels==1));
% histfit(temp(:))

temp = [];
for i = 0:4
temp = [temp, D1(find(labels==i), find(labels~=i))];
end

histogram(temp(:), 50, 'Normalization','probability')

legend('within-class', 'between-class')
title('')
figure()
