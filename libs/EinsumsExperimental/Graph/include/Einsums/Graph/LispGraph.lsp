;;----------------------------------------------------------------------------------------------
;; Copyright (c) The Einsums Developers. All rights reserved.
;; Licensed under the MIT License. See LICENSE.txt in the project root for license information.
;;----------------------------------------------------------------------------------------------

;; The C++ metaprogramming system is very similar to Lisp.
;; Therefore, this program can be used as an example for how to implement this at compile time.

;; Nodes will be some unique identifier, like a string.
;; Edges will be a four-part list.
(defstruct graph
    nodes
    edges
)

(defstruct edge
    start
    directed
    end
    weight
)

(defun traverse (edge from) "Traverse an edge if possible."
    (if (= (edge-start edge) from)
        (edge-end edge)
        (if (and (= (edge-directed edge) "-") (= (edge-end edge) from))
            (edge-start edge)
            nil
        )
    )
)

(defun can-traverse (edge from) "Sees if an edge can be traversed from the input."
    (or (= (edge-start edge) from) (and (= (edge-directed edge) "-") (= (edge-end edge) from)))
)

(defun add-node (graph node) "Adds a node to the graph."
    (nconc (graph-nodes graph) '(node))
)

(defun add-edge (graph edge) "Adds an edge to the graph."
    (nconc (graph-edges graph) '(edge))
)

;; For reference, nconc is similar to std::tuple_cat.

