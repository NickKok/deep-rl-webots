function mymap = drawpoint(mymap,P,r)

    PN = P./mymap.size;
    pCN = mymap.center + PN;
    pC = pCN.*mymap.size;

    xRange = round(pC(1)-r:1:pC(1)+r);
    yRange = round(pC(2)-r:1:pC(2)+r);

    [xNew,yNew] = meshgrid(xRange,yRange);
    mymap.map(xNew,yNew) = 255;
end