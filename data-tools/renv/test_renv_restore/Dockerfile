FROM rocker/r-base:4.0.2
# install renv package
RUN Rscript -e "install.packages('renv')"
# copy everything to docker, including renv.lock file
COPY . /app
# set working directory
WORKDIR /app
# restore all the packages
RUN Rscript -e "renv::restore()"
# run our R code
CMD ["Rscript", "main.R"]