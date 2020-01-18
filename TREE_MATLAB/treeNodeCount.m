function [nodecount, leafcount] = treeNodeCount(tree)
    if ~isempty(tree.kids)
        [lnodecount, lleafcount] = treeNodeCount(tree.kids{1,1});
        [rnodecount, rleafcount] = treeNodeCount(tree.kids{1,2});
        nodecount = 1 + lnodecount + rnodecount;
        leafcount = lleafcount + rleafcount;
    else
        nodecount = 0;
        leafcount = 1;
    end
end