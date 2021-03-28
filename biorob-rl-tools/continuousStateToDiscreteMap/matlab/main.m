mymap.map = zeros(200,200);
mymap.size = [200,200];
mymap.center = [0.3,0.5];

% Schema
% [torso,hips,knees,ankles,foots]

points = [
    0,0
    30,0
    60,20
    60,-15
    90,10
    90,-30
    90,5
    90,-35
    ];

pointsFromWebots = [
    1.4316434391768058, 0.2652967474702148
    1.041002046812994, 0.17931899279901475
    0.5551639712372729, 0.06120646336175953
    0.12456667052413116, -0.19289200172290702
    1.0409887605138592, 0.17932448748154078
    0.5827596631783166, 0.37939026680981286
    0.1988565214376072, 0.059046221424305954
];

% convert in centimeters and adapt axis and reference
pointsFromWebots = 100*(pointsFromWebots - repmat(pointsFromWebots(1,:),7,1));
pointsFromWebots(:,1) = pointsFromWebots(:,1)*-1;

r = 3;


pointsTestWebotsNew = reshape(pointsTestWebots,size(pointsTestWebots,1),size(pointsTestWebots,2)/7,size(pointsTestWebots,2)/3);
hold on;
set(gcf,'DoubleBuffer','on') % To turn it on
for i = 1:size(pointsTestWebotsNew,1)
    mymap.map = zeros(200,200);
    myPoints = reshape(pointsTestWebotsNew(i,[2,3],:),2,7)';
    % convert in centimeters and adapt axis and reference
    myPoints = 100*(myPoints - repmat(myPoints(1,:),7,1));
    myPoints(:,1) = myPoints(:,1)*-1;
    for p = 1:length(myPoints)
        mymap = drawpoint(mymap,myPoints(p,:),1.5);
    end
    %pause(0.01)
    imshow(mymap.map)
    drawnow;
end

%for p = 1:length(pointsFromWebots)
%    mymap = drawpoint(mymap,pointsFromWebots(p,:),1.5);
%end



