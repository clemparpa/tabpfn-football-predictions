By registering for and submitting to this competition, you confirm that you have read, understood, and accepted these rules in full.
 
1. Overview
This is a free-to-enter forecasting competition in which participants predict the outcomes of football matches. Predictions are scored against actual results using log-loss and ranked on a public leaderboard. Lower multi-class log-loss is better. The competition is for learning and reputation only: there is no entry fee, no purchase is necessary.
 
2. Eligibility & registration
•
The competition is open to individuals who are at least 18 years old at the time of registration.
•
A valid user account is required. All submissions are tied to your account. You must provide accurate registration information and you may only register one account for yourself; accounts are not transferable.
•
During sign-up you must create an account and accept these rules. Any personal data you provide is processed in accordance with the Prior Labs Privacy Policy (https://priorlabs.ai/privacy-policy).
•
You must choose a public nickname (see §7).
3. Phases & deadlines
•
The competition runs in rounds/phases. Submissions open and close at set times for each round.
•
For any given bracket, submissions close 8 hours before that bracket's scheduled kickoff. Late submissions are not accepted.
•
Result submissions open: 27 June.
4. What you are predicting
•
For each match you submit probabilities for the three possible outcomes: home win, draw, home loss.
•
Outcomes are decided on the 90-minute (regulation) result only. Extra time and penalty shootouts do not count toward the outcome. For example, a match level at 1–1 after 90 minutes that is then decided on penalties is scored as a draw.
5. Submission format
•
A submission is a single CSV file that must strictly conform to the schema provided in the repository. The schema and a sample file (headers plus example prediction rows) are available there.
•
On upload, submissions are validated. Your file must use valid home/away team codes, contain no empty or missing required fields, and give each outcome a probability strictly greater than 0 and strictly less than 1.
•
You will be warned if your file is missing matches, or if a match's three outcome probabilities do not sum to 1.
6. Modeling requirements
•
Predictions must be generated using TabPFN.
•
You may combine TabPFN with other models, both as an ensemble member and via stacking, provided TabPFN is part of your pipeline.
•
Reproducibility. You must provide a link to a repository containing your code; ensure that, given the same input data, your code reproduces your submitted predictions; and include your input data in the repo, or make it available at a public URL.
•
A boilerplate repo is provided as a starting point.
•
Your responsibility for code and data. You are solely responsible for the code, data, and other materials you submit or publish in connection with the competition. You represent and warrant that you own or hold all rights necessary to use and publish them and that they do not infringe any third party's rights. Prior Labs does not review, endorse, or assume any responsibility for participant-submitted code or data.
•
Use of submissions. Prior Labs does not intend to use participants' submitted predictions or code in its products or services.
•
API credits for participants. The Prior Labs API (including its "thinking" mode) is free for everyone, with a limited default usage allowance. When you register for the competition and submit a request, Prior Labs will grant you additional API credits. These credits are provided free of charge and cannot be purchased, have no cash value, are non-transferable, and may only be used with the Prior Labs API subject to the applicable Terms and Acceptable Use Policy. They are not a prize and do not depend on your leaderboard ranking. The amount and conditions are at Prior Labs' fair, reasonable, and non-discriminatory discretion and may change.
7. Leaderboard & nicknames
•
Your nickname is displayed publicly on the leaderboard alongside your score. By participating, you consent to this public display.
•
Nicknames must be safe-for-work. Participants with nicknames deemed offensive may be removed at the organizers' discretion.
8. Scoring
•
Scoring metric: multi-class log-loss (lower is better).
•
Leaderboard ranking is by average multi-class log-loss across all matches in scope.
•
Missing submissions for a match are scored as a uniform prediction of 0.33 / 0.33 / 0.33 across the three outcomes.
•
Authoritative results and excluded matches. Outcomes are determined by the organizers based on the official 90-minute result. Matches that are abandoned, postponed, rescheduled outside the competition window, replayed, awarded/forfeited off the pitch, or otherwise disputed or undetermined are excluded from scoring. The organizers' determination of results and of which matches are in scope is final.
9. Prize — Interview Invitation
•
The participant with the lowest average multi-class log-loss score across all matches in scope at the close of the competition is the "Winner" and is entitled to be invited to a job interview for an open data science position at Prior Labs GmbH, bypassing the initial HR screening stage (the "Prize").
•
If two or more participants share the lowest score, the Prize is awarded to the participant among them whose final qualifying submission was submitted earliest. Prior Labs will notify the Winner by message to the registered account within 14 days of the competition closing.
•
The Winner must confirm acceptance of the Prize within 14 days of notification. If the Winner does not respond within that period or declines, the entitlement lapses and Prior Labs may, at its discretion, offer the Prize to the next eligible participant.
•
The Prize grants access to the interview stage only. The Winner is assessed under the same criteria and process applied to all other candidates at that stage and receives no preferential treatment.
•
Participation in the competition confers no right to employment.
•
Prior Labs reserves the right to withdraw or modify the Prize if, at the time the Winner is determined, no suitable open data science position exists at Prior Labs GmbH, or if the Winner does not meet the minimum eligibility requirements applicable to all candidates for that position. In such case, no alternative prize or compensation is owed.
10. Data protection & terms
•
The data controller for personal data processed in connection with this competition is Prior Labs GmbH, Kaiser-Joseph-Str. 254, 79098 Freiburg, Germany.
•
In addition to the Prior Labs Privacy Policy (https://priorlabs.ai/privacy-policy), Prior Labs processes your nickname, submissions, and scores to operate the competition and display the public leaderboard (Art. 6(1)(b) GDPR). If you are the Winner, your contact details and interview-related information are processed to fulfil the Prize (Art. 6(1)(b) GDPR). If you have separately consented to marketing communications, your e-mail address is processed on that basis (Art. 6(1)(a) GDPR); you may withdraw that consent at any time via dataprotection@priorlabs.ai.
•
These rules supplement the Prior Labs General Terms and Conditions (https://priorlabs.ai/general-terms-and-conditions) and Acceptable Use Policy (https://priorlabs.ai/aup). In the event of a conflict regarding the competition specifically, these competition rules prevail.
•
The competition is governed by the laws of the Federal Republic of Germany. For participants who are consumers habitually resident in another country, the choice of German law does not deprive them of the protection afforded by the mandatory provisions of the law of their country of habitual residence (Art. 6(2) Rome I Regulation).
•
If the participant is a merchant (Kaufmann), a legal entity under public law, or a special fund under public law within the meaning of the German Code of Civil Procedure, the exclusive place of jurisdiction for all disputes arising from or in connection with the competition is Berlin, Germany. For all other participants, the statutory rules on jurisdiction apply.
•
Prior Labs provides the competition platform, infrastructure, and leaderboard free of charge. For this unremunerated provision, Prior Labs is liable only for damages caused by intent (Vorsatz) or gross negligence (grobe Fahrlässigkeit), in accordance with § 521 BGB applied by analogy. Liability for slight negligence (leichte Fahrlässigkeit) in connection with the platform is excluded to the fullest extent permitted by law. Regarding the Prize awarded under § 657 BGB, Prior Labs is liable without limitation for damages caused by intent or gross negligence, and for damages arising from injury to life, body, or health. For damages caused by slight negligence in connection with the Prize, Prior Labs is liable only where a material contractual obligation (Kardinalpflicht) has been breached, limited to the foreseeable and typical damage. In all cases, liability under the German Product Liability Act (Produkthaftungsgesetz) and for fraudulent misrepresentation remains unaffected. These limitations apply equally to Prior Labs' legal representatives and vicarious agents.
•
The organizers may disqualify any participant who breaches these rules or attempts to manipulate the competition, and may amend, suspend, or cancel the competition where there is a valid reason to do so considering participants' reasonable interests.