/**
 * markPhase1Task1Complete
 *
 * Marks row 4 ("Git repo + environment setup") as complete:
 *   - fills columns A–C green with white text
 *   - fills the Gantt bar cells (D onward) with a darker done-green
 *   - prepends "✓ " to the task name in column B
 *
 * HOW TO RUN:
 *   Extensions → Apps Script → paste → Run → markPhase1Task1Complete
 */

function markPhase1Task1Complete() {
  const ss      = SpreadsheetApp.getActiveSpreadsheet();
  const sheet   = ss.getActiveSheet();
  const numCols = sheet.getLastColumn();

  const TASK_ROW   = 4;          // Row 4 = "Git repo + environment setup"
  const LABEL_COL  = 2;          // Column B = task name
  const GANTT_START = 4;         // Column D = first week column

  const DONE_BG_LABEL = '#2e7d32';   // dark green for label columns (A–C)
  const DONE_BG_BAR   = '#4caf50';   // mid green for Gantt bar cells
  const DONE_FONT     = '#ffffff';

  // ── Colour label columns A–C ─────────────────────────────────────
  sheet.getRange(TASK_ROW, 1, 1, 3)
    .setBackground(DONE_BG_LABEL)
    .setFontColor(DONE_FONT);

  // ── Colour Gantt bar columns D onward ────────────────────────────
  sheet.getRange(TASK_ROW, GANTT_START, 1, numCols - GANTT_START + 1)
    .setBackground(DONE_BG_BAR)
    .setFontColor(DONE_FONT);

  // ── Prepend ✓ to the task label in column B ───────────────────────
  const labelCell = sheet.getRange(TASK_ROW, LABEL_COL);
  const existing  = String(labelCell.getValue());
  if (!existing.startsWith('✓')) {
    labelCell.setValue('✓ ' + existing);
  }

  SpreadsheetApp.getUi().alert('✅ Phase 1 — Task 1 marked as complete.');
}
