/* Progressive enhancement for the parlay analyzer: add more leg rows.
   The page is fully usable without JS — this only adds convenience. */
(function () {
  "use strict";
  var addBtn = document.getElementById("add-leg");
  var body = document.getElementById("legs-body");
  if (!addBtn || !body) return;

  addBtn.addEventListener("click", function () {
    var count = body.querySelectorAll("tr").length;
    var tr = document.createElement("tr");
    tr.innerHTML =
      '<td><label class="visually-hidden" for="label' + count + '">Leg ' + (count + 1) + ' name</label>' +
      '<input id="label' + count + '" name="label" type="text" placeholder="e.g. Team ML"></td>' +
      '<td><label class="visually-hidden" for="prob' + count + '">Leg ' + (count + 1) + ' probability</label>' +
      '<input id="prob' + count + '" name="prob" type="number" step="0.01" min="0" max="1" placeholder="0.60"></td>' +
      '<td><label class="visually-hidden" for="odds' + count + '">Leg ' + (count + 1) + ' decimal odds</label>' +
      '<input id="odds' + count + '" name="decimal_odds" type="number" step="0.01" min="1" placeholder="1.80"></td>' +
      '<td><label class="visually-hidden" for="game' + count + '">Leg ' + (count + 1) + ' game ID</label>' +
      '<input id="game' + count + '" name="game_id" type="text" placeholder="auto"></td>';
    body.appendChild(tr);
    tr.querySelector("input").focus();
  });
})();
