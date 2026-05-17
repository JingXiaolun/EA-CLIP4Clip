# clean file endsup with 'swp' and __pycache__ directory
find . -name *swp|xargs rm -rf {}
find . -name __pycache__|xargs rm -rf {}
