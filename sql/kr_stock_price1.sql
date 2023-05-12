select A.TD AS DATE
, 'A'+A.code AS CODE
, B.KN  AS NAME
, A.OPEN_P
, A.HIGH_P
, A.LOW_P
, A.CLOSE_AP
from MARKET..MB A
, market..CA B
where 1=1
and A.TD > '{from_date}'
and A.CODE in ('069500','102110','114800','130730','148020','196230','272580')
and A.CODE = B.CODE
