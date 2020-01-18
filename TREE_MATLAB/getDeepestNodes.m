function nodes = getDeepestNodes(tree)
    if isempty(tree.kids)
        nodes = [];    
    elseif isempty(tree.kids{1,1}.kids) && isempty(tree.kids{1,2}.kids)
        nodes = tree;
    else
        nodesl = getDeepestNodes(tree.kids{1,1});
        nodesr = getDeepestNodes(tree.kids{1,2});
        nodes = [nodesl, nodesr];
    end
end