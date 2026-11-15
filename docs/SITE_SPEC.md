# Site design spec

What the best hardware and robot company sites do, and what DroneBench should take from them.
Looked at on 2026-11-14 in a real browser: Figure, 1X, Zipline, Skydio, Anduril, ARC Prize, and Cosmos
searches for "robotics", "drone", "exploded view technical diagram" and "hardware startup website".

## What the good ones share

1. **The product fills the first screen.** Figure is one pale studio shot of the robot cropped large. 1X shows three
   heads side by side. Zipline and Skydio are full-bleed footage of the real aircraft. Nothing sits next to the
   object that competes with it.
2. **Almost no words up top.** "LOOK UP" (Zipline). "Fast. Far. Autonomous." (Skydio). "NEO Home Robot" with the
   second word in grey (1X). One sentence, one or two buttons, and that is the whole hero.
3. **One type voice, used at two extreme sizes.** A huge bold headline, a small plain line under it, nothing in
   between. Anduril sets everything in wide uppercase. Zipline uses an extra-condensed heavy face. Skydio a
   heavy grotesk.
4. **Buttons are few and flat.** Pill or outline, sentence case, never more than two visible together.
   A solid dark pill for the one action that matters (1X "Order", Skydio "Contact").
5. **Proof is a number or a chart, not a paragraph.** ARC Prize shows a score-against-cost chart and "$2M" in pixel
   type. The leaderboard is the product.
6. **Colour is the object plus one accent.** Cool grey or warm off-white behind a physical thing; black pages
   only when the footage is dark. The accent appears on the button or one tag, not everywhere.
7. **The page moves because the product moves.** Footage, turning models, scroll-driven views of the thing itself.
   Decoration that moves for its own sake is absent.
8. **Diagrams are drawn, not stock.** On Cosmos the strongest technical visuals are exploded views: thin line art
   (blue on grey, white on black), numbered callouts, one object pulled apart. The weakest hits for
   "hardware startup website" were generic mockups with purple gradients, glass cards and floating phones. That
   is the look to avoid.

## Sites opened for this spec (2026-11-14, real browser, desktop 1440 wide)

First screen of each, plus a scroll down the page for 1X, Zipline and ARC Prize:
https://www.figure.ai, https://www.1x.tech, https://www.zipline.com, https://www.skydio.com, https://www.anduril.com, https://arcprize.org.
Cosmos (https://www.cosmos.so, signed in): element searches "robotics", "drone", "exploded view technical diagram", "hardware startup website".
Not opened: Boston Dynamics, Apptronik, Joby, Wing. Web searches for "best hardware startup websites" returned rankings of companies, not of their sites, so nothing was taken from them.

## Cosmos pages opened for the blueprint rebuild (2026-11-15, signed in as Adam)

His own boards (https://www.cosmos.so/sisiphamus/collections: "life os", "Pitch Deck", "poster"), and element searches "blueprint technical drawing poster" and "line art airplane wireframe". What they show: saturated cyanotype blue sheets, white line art, title blocks, dimension lines, huge white condensed type (the Pitch Deck board). The final page takes the blue sheet, the white line drawing, the dimension line and the title block from these.

## What this means for DroneBench

DroneBench has no product photography and should not fake any. What it has is a drone that exists only as geometry and
a test that flies it. So the object on the page is the 3D model, and the story is the thing no other site can show:
**a design changing as an agent learns from the tests.**

The current example barely changes (the Falcon starts almost finished, so the agent only shaves weight). A page about
iteration needs a run where the shape visibly changes. The Sonnet-from-nothing run (see `runs/scratch-sonnet`) is
that run, and the history section should be rebuilt from it once it finishes.

## Page, top to bottom

1. **Hero.** The final design, huge, on cool pale grey (#e9eaee), slowly turning, lit like a studio shot. Left
   bottom: a headline of at most seven words and one plain sentence. Two buttons: "Test your agent" (solid dark),
   "Watch it design" (outline). Two nav links.
2. **The test, as an exploded drawing.** One line drawing of the final drone pulled apart, drawn from its real
   parts, with the checks and the numbers the drone got listed beside it. Callouts pinned to parts were tried and
   dropped: the parts that decide a check sit too close together to label cleanly. Blue line art on a grey panel.
3. **History (the signature).** Scroll scrubs the 3D model from the first passing design to the last, parts that changed drawn blue, the body see-through so the battery shows.
   A running score and one line per step. Failed attempts appear as short ticks, grouped. Starts from a blank
   design, so the first frames are very different from the last.
4. **Race.** Best score so far against attempts, one line per agent, labelled directly (ARC Prize style chart on the
   page colour, no legend box). Under it, one honest line: one run each, not a ranking.
5. **Bring your agent.** The install and run commands with a copy button, on the committed blue. Nothing else.
6. **What it does not know.** Three limits, plain text on near-black.

## Type, colour, spacing

- Typeface: Archivo (already self-hosted), using its width axis: extra-condensed heavy for the hero word and
  section numbers, normal width for everything else. Two sizes dominate; everything else is body.
- Colour: pale grey #e9eaee page, near-black #0a0b1a text, one blue #2a3cff for buttons, the changed parts and the
  closing section. Orange is gone. Black pages only for the limits block.
- Spacing: 4 point scale, 96 px between sections, body copy at most 65 characters wide.
- Buttons: pill, sentence case, 44 px minimum height, visible keyboard focus.

## What not to do

- No gradients, glass cards, icon tiles, grids as wallpaper, or tiny tracked labels in corners.
- No claim without a number on the page. No stock photo of a drone that is not this drone.
- Nothing that moves unless the drone moves.
